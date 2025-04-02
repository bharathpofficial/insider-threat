export interface OriginalData {
  date: string;
  user: number;
  pc: number;
  to_outside: boolean;
  from_outside: boolean;
  has_attachment: boolean;
  is_exe: boolean;
  file_open: boolean;
  file_write: boolean;
  file_copy: boolean;
  file_delete: boolean;
  connected: boolean;
  logon: boolean;
}

export interface ThreatData {
  timestamp: string;
  is_anomaly: boolean;
  mse: number;
  original_data: OriginalData;
}

export interface ApiResponse {
  status: string;
  data: ThreatData[];
} 