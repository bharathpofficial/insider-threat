import { ApiResponse, ThreatData } from '../types/types';

const API_BASE_URL = 'http://localhost:8000';

const isValidData = (data: any): data is ThreatData[] => {
  return (
    Array.isArray(data) &&
    data.every(item => 
      item &&
      typeof item === 'object' &&
      'timestamp' in item &&
      'original_data' in item &&
      item.original_data !== null
    )
  );
};

export const fetchThreats = async (): Promise<ApiResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/anomalies/threats`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  } catch (error) {
    console.error('Error fetching threats:', error);
    return { status: 'error', data: [] };
  }
};

export const fetchLatestData = async (): Promise<ApiResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/anomalies/latest`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  } catch (error) {
    console.error('Error fetching latest data:', error);
    return { status: 'error', data: [] };
  }
};

export const subscribeToUpdates = (onData: (data: ThreatData[]) => void) => {
  let eventSource: EventSource | null = null;
  let reconnectAttempt = 0;
  const maxReconnectAttempts = 5;

  const connect = () => {
    if (eventSource) {
      eventSource.close();
    }

    try {
      console.log('Connecting to SSE...');
      eventSource = new EventSource(`${API_BASE_URL}/anomalies/stream`);

      eventSource.onopen = () => {
        console.log('SSE connection opened');
        reconnectAttempt = 0;
      };

      eventSource.onmessage = (event) => {
        if (event.data === "") return; // Ignore empty ping messages
        
        try {
          const response = JSON.parse(event.data);
          
          if (response.status === 'success' && isValidData(response.data)) {
            console.log('Received new data:', response.data.length);
            onData(response.data);
          } else if (response.status === 'error') {
            console.error('Server error:', response.message);
          }
        } catch (error) {
          console.error('Error parsing SSE data:', error);
        }
      };

      eventSource.onerror = (error) => {
        console.error('SSE Error:', error);
        eventSource?.close();
        
        if (reconnectAttempt < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempt), 10000);
          console.log(`Attempting to reconnect in ${delay}ms...`);
          
          setTimeout(() => {
            reconnectAttempt++;
            connect();
          }, delay);
        } else {
          console.error('Max reconnection attempts reached');
        }
      };

    } catch (error) {
      console.error('Error setting up SSE:', error);
    }
  };

  connect();

  return () => {
    if (eventSource) {
      console.log('Cleaning up SSE connection');
      eventSource.close();
      eventSource = null;
    }
  };
}; 