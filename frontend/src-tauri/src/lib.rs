use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::Manager;

// Store the Python process handle
struct PythonBackend {
  #[allow(dead_code)]
  process: Option<Child>,
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .setup(|app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }

      // Spawn Python backend process
      let python_backend = start_python_backend();
      app.manage(Mutex::new(python_backend));

      Ok(())
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}

fn start_python_backend() -> PythonBackend {
  // Get the directory where the binary is running
  let backend_path = if cfg!(debug_assertions) {
    // During development, use the backend directory
    std::path::PathBuf::from("../../backend")
  } else {
    // During production, look for backend next to the app
    std::path::PathBuf::from("../backend")
  };

  match Command::new("python3")
    .arg("app.py")
    .current_dir(backend_path.clone())
    .spawn() {
    Ok(process) => {
      log::info!("Python backend started successfully");
      PythonBackend {
        process: Some(process),
      }
    }
    Err(e) => {
      log::warn!("Failed to start Python backend: {}. Make sure Python is installed and backend/app.py exists.", e);
      PythonBackend { process: None }
    }
  }
}
