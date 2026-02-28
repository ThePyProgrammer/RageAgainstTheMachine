import { useEffect, useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import type { GLTF } from "three-stdlib";
import * as THREE from "three";
import type { ElectrodeMapping, StreamStatus } from "@/types/eeg";
import { getEEGColor, updateHistoryBuffer } from "@/utils/brainHeatmap";
import { STANDARD_ELECTRODES } from "@/config/eeg";

interface BrainModelProps {
  electrodeData?: Record<string, number>;
  mapping?: ElectrodeMapping;
  status?: StreamStatus;
}

// Data structure for pre-calculated influences
// Each vertex stores a list of: [ElectrodeIndex, NormalizedWeight, ElectrodeIndex, NormalizedWeight...]
type VertexInfluence = number[];

const BrainModel = ({
  electrodeData = {},
  mapping = {},
  status = "disconnected",
}: BrainModelProps) => {
  const gltf = useGLTF("/3d/bci/brain.glb") as GLTF;
  const groupRef = useRef<THREE.Group>(null!);

  // Store the pre-calculated influence map
  // This replaces the heavy "weights" array
  const influenceMapRef = useRef<Map<THREE.Mesh, VertexInfluence[]>>(new Map());

  const historyRef = useRef<Record<string, number[]>>({});
  const frameCountRef = useRef(0);

  // Cache standard electrode IDs for fast lookup
  const electrodeIds = useMemo(() => STANDARD_ELECTRODES.map((e) => e.id), []);

  useEffect(() => {
    // 1. Setup Model Transforms
    const box = new THREE.Box3().setFromObject(gltf.scene);
    const center = box.getCenter(new THREE.Vector3());
    gltf.scene.position.sub(center);
    gltf.scene.scale.set(0.27, 0.27, 0.27);
    gltf.scene.position.setY(gltf.scene.position.y + 0.73);
    gltf.scene.position.setZ(gltf.scene.position.z - 0.13);
    gltf.scene.rotation.set(0, -Math.PI / 2, 0);
    gltf.scene.updateMatrixWorld(true); // Crucial for worldToLocal

    // 2. Pre-calculate Influences
    gltf.scene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;
        const geometry = mesh.geometry;

        // Initialize colors if missing
        if (!geometry.attributes.color) {
          const count = geometry.attributes.position.count;
          geometry.setAttribute(
            "color",
            new THREE.BufferAttribute(new Float32Array(count * 3), 3),
          );
        }

        mesh.material = new THREE.MeshStandardMaterial({
          vertexColors: true,
          metalness: 0.3,
          roughness: 0.5,
        });
        mesh.castShadow = true;
        mesh.receiveShadow = true;

        // A. Calculate Bounds & Radius
        geometry.computeBoundingBox();
        const size = new THREE.Vector3();
        geometry.boundingBox!.getSize(size);
        const avgSize = (size.x + size.y + size.z) / 3;
        const MAX_DISTANCE = avgSize * 0.7; // Wide influence for smooth gradients
        const FALLOFF = 1.3; // Gentle falloff for natural blending

        // B. Map Electrodes to Local Space
        const localElectrodes = STANDARD_ELECTRODES.map((e) =>
          mesh.worldToLocal(new THREE.Vector3(e.x, e.y, e.z)),
        );

        const positions = geometry.attributes.position.array as Float32Array;
        const influences: VertexInfluence[] = [];

        // C. Build Sparse Influence Map
        // Instead of storing 0s for far-away electrodes, we only store the relevant ones.
        for (let i = 0; i < positions.length / 3; i++) {
          const vPos = new THREE.Vector3(
            positions[i * 3],
            positions[i * 3 + 1],
            positions[i * 3 + 2],
          );

          let totalWeight = 0;
          const vertexInfluences: number[] = []; // Pairs of [Index, Weight]

          // Check distance to every electrode
          localElectrodes.forEach((ePos, eIdx) => {
            const dist = vPos.distanceTo(ePos);

            if (dist < MAX_DISTANCE) {
              // Inverse Distance Weighting
              const weight = 1 / Math.pow(dist + avgSize * 0.05, FALLOFF);
              vertexInfluences.push(eIdx, weight);
              totalWeight += weight;
            }
          });

          // Normalize weights so they sum to 1.0
          // This removes the need for division in the animation loop!
          if (totalWeight > 0) {
            for (let j = 1; j < vertexInfluences.length; j += 2) {
              vertexInfluences[j] /= totalWeight;
            }
          }

          influences.push(vertexInfluences);
        }

        influenceMapRef.current.set(mesh, influences);
      }
    });
  }, [gltf, electrodeIds]);

  useFrame(() => {
    frameCountRef.current++;
    const isConnected = status === "connected";
    const hasMappedElectrodes = Object.keys(mapping).length > 0;

    // 1. Prepare Data Buffer (Fast Array)
    // Map electrode IDs to a flat array of current values
    // Index matches STANDARD_ELECTRODES order
    const valuesBuffer = new Float32Array(STANDARD_ELECTRODES.length);

    if (isConnected && hasMappedElectrodes) {
      STANDARD_ELECTRODES.forEach((electrode, idx) => {
        const channelId = mapping[electrode.id];
        if (channelId) {
          const raw = electrodeData[`fch${channelId}`] ?? 0;
          if (!historyRef.current[electrode.id])
            historyRef.current[electrode.id] = [];

          const smoothed = updateHistoryBuffer(
            historyRef.current[electrode.id],
            raw,
            10,
          );
          valuesBuffer[idx] = smoothed;
        } else {
          valuesBuffer[idx] = 0;
        }
      });
    }

    // 2. Update Colors using Sparse Map
    // We reuse one Color object to avoid garbage collection
    // const tempColor = new THREE.Color()
    const baselineColor = getEEGColor(0);
    const darkGray = new THREE.Color(0.3, 0.3, 0.3);

    influenceMapRef.current.forEach((influences, mesh) => {
      const colorAttr = mesh.geometry.attributes.color as THREE.BufferAttribute;

      for (let i = 0; i < influences.length; i++) {
        const vertexInf = influences[i];

        if (!isConnected) {
          colorAttr.setXYZ(i, darkGray.r, darkGray.g, darkGray.b);
          continue;
        }

        if (vertexInf.length === 0) {
          // Vertex has no nearby electrodes -> Baseline
          colorAttr.setXYZ(
            i,
            baselineColor.r,
            baselineColor.g,
            baselineColor.b,
          );
          continue;
        }

        // Calculate weighted sum (Dot Product)
        let finalValue = 0;
        for (let k = 0; k < vertexInf.length; k += 2) {
          const eIdx = vertexInf[k];
          const weight = vertexInf[k + 1];
          finalValue += valuesBuffer[eIdx] * weight;
        }

        // Apply Color
        const c = getEEGColor(finalValue);
        colorAttr.setXYZ(i, c.r, c.g, c.b);
      }
      colorAttr.needsUpdate = true;
    });
  });

  return (
    <group ref={groupRef}>
      <primitive object={gltf.scene} />
    </group>
  );
};

useGLTF.preload("/3d/bci/brain.glb");

export default BrainModel;
