import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import DashboardLayout from "@/layouts/DashboardLayout";
import { routes } from "@/config/routes";
import { ElectrodeMappingProvider } from "@/contexts/ElectrodeMappingContext";
import { SessionStatusProvider } from "@/contexts/SessionStatusContext";
import { DeviceProvider } from "@/contexts/DeviceContext";

function App() {
  return (
    <DeviceProvider>
      <SessionStatusProvider>
        <ElectrodeMappingProvider>
          <Router>
            <Routes>
              <Route element={<DashboardLayout />}>
                {routes.map(({ path, element: Element }) => (
                  <Route key={path} path={path} element={<Element />} />
                ))}
              </Route>
            </Routes>
          </Router>
        </ElectrodeMappingProvider>
      </SessionStatusProvider>
    </DeviceProvider>
  );
}

export default App;
