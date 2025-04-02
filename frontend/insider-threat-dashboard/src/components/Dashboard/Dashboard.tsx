import React, { useEffect, useState, useCallback, useRef } from 'react';
import { ThreatData } from '../../types/types';
import { subscribeToUpdates } from '../../services/api';
import ThreatTable from '../ThreatTable/ThreatTable';
import './Dashboard.css';

const THROTTLE_INTERVAL = 1000; // 1 second throttle

const Dashboard: React.FC = () => {
  const [data, setData] = useState<ThreatData[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdateTime, setLastUpdateTime] = useState<Date | null>(null);
  const lastUpdateRef = useRef<number>(0);

  const updateData = useCallback((newThreats: ThreatData[]) => {
    const now = Date.now();
    
    // Throttle updates
    if (now - lastUpdateRef.current < THROTTLE_INTERVAL) {
      return;
    }
    
    lastUpdateRef.current = now;
    
    setData(currentData => {
      // Create a map of existing data using timestamp as key
      const existingDataMap = new Map(
        currentData.map(item => [item.timestamp, item])
      );

      // Add new threats that don't already exist
      newThreats.forEach(threat => {
        if (!existingDataMap.has(threat.timestamp)) {
          existingDataMap.set(threat.timestamp, threat);
        }
      });

      // Convert map back to array and sort by timestamp
      const updatedData = Array.from(existingDataMap.values())
        .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

      setLastUpdateTime(new Date());
      setIsConnected(true);
      return updatedData;
    });
  }, []);

  useEffect(() => {
    setIsConnected(false);
    const unsubscribe = subscribeToUpdates(updateData);

    return () => {
      unsubscribe();
    };
  }, [updateData]);

  return (
    <div className="dashboard">
      <h1>Real-time Insider Threat Detection</h1>
      <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
        {isConnected ? '🟢 Live Updates Active' : '🔴 Connection Lost'} 
        ({data.length} items)
        {lastUpdateTime && (
          <span className="last-update">
            Last update: {lastUpdateTime.toLocaleTimeString()}
          </span>
        )}
      </div>
      <ThreatTable data={data} />
    </div>
  );
};

export default Dashboard; 