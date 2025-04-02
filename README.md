# ![Favicon](frontend/insider-threat-dashboard/src/assets/insider.ico) insider-threat detection
ML based, using LSTM and AutoEncoders for training on set of features on normal dataset. Later, finding anamoly that is deviating from this normal pattern, marked as threat.

# Run this Project
Do the necessary git repo cloning and `cd insider-threat` dir.
- ## Run the Backend Server
  ```bash
  # assuming from project root dir 
  cd PROJECT/API
  python server.py
  ```
- ## Run Email Server
  ```bash
  # assuming from project root dir
  cd frontend/insider-threat-dashboard/src/components/Email
  node emailServer.cjs
  ```
- ## Run Frontend
  ```bash
  # assuming from project root dir
  cd frontend/insider-threat-dashboard
  npm run dev
  ```
  
  
