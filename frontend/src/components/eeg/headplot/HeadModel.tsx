import { useEffect, useRef } from "react";
import { useGLTF } from "@react-three/drei";
import type { GLTF } from "three-stdlib";
import * as THREE from "three";

const HeadModel = () => {
  const gltf = useGLTF("/3d/bci/head.glb") as GLTF;
  const groupRef = useRef<THREE.Group>(null!);

  useEffect(() => {
    const box = new THREE.Box3().setFromObject(gltf.scene);
    const center = box.getCenter(new THREE.Vector3());
    gltf.scene.position.sub(center);

    // Scale up the head slightly to prevent brain clipping
    gltf.scene.scale.set(1.1, 1.13, 1.1);

    gltf.scene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;

        if (mesh.geometry) {
          mesh.geometry.computeVertexNormals();
        }

        mesh.material = new THREE.MeshStandardMaterial({
          color: 0x888888,
          transparent: true,
          opacity: 0.5,
          metalness: 0.2,
          roughness: 0.6,
          side: THREE.DoubleSide,
        });
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.renderOrder = 2;
      }
    });
  }, [gltf]);

  return (
    <group ref={groupRef}>
      <primitive object={gltf.scene} />
    </group>
  );
};

useGLTF.preload("/3d/bci/head.glb");

export default HeadModel;
