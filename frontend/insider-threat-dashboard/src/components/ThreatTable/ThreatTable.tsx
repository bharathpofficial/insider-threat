import React, { useEffect, useRef, useState, useCallback } from 'react';
import { ThreatData } from '../../types/types';
import Modal from '../Modal/Modal';
import ToggleSwitch from '../ToggleSwitch/ToggleSwitch';
import {  sendEmailNotification } from '../../services/emailService';
import './ThreatTable.css';

interface ThreatTableProps {
  data: ThreatData[];
}

const ThreatTable: React.FC<ThreatTableProps> = ({ data }) => {
  const [autoScroll, setAutoScroll] = useState(true);
  const tableRef = useRef<HTMLDivElement>(null);
  const isUserScrolling = useRef(false);
  const lastScrollTop = useRef(0);
  const scrollTimeout = useRef<NodeJS.Timeout | null>(null);
  const dataLengthRef = useRef(data.length);
  const [selectedThreat, setSelectedThreat] = useState<ThreatData | null>(null);
  const [isAutoEmail, setIsAutoEmail] = useState(false);

  // Check for new data and handle automatic email
  useEffect(() => {
    const hasNewData = data.length !== dataLengthRef.current;
    
    if (hasNewData) {
      const newThreat = data[data.length - 1];
      // Send email automatically if auto-email is enabled and it's a threat
      if (isAutoEmail && newThreat && newThreat.is_anomaly) {
        console.log('New threat detected, sending automatic email:', newThreat);
        sendEmailNotification(newThreat)
          .then(() => console.log('Auto email notification sent'))
          .catch(error => console.error('Failed to send auto email:', error));
      }
    }

    dataLengthRef.current = data.length;

    if (hasNewData && autoScroll && !isUserScrolling.current) {
      requestAnimationFrame(() => {
        if (tableRef.current) {
          tableRef.current.scrollTop = tableRef.current.scrollHeight;
        }
      });
    }
  }, [data, autoScroll, isAutoEmail]);

  // Handle scroll events
  const handleScroll = useCallback(() => {
    if (!tableRef.current) return;

    if (scrollTimeout.current) {
      clearTimeout(scrollTimeout.current);
    }

    const container = tableRef.current;
    const currentScrollTop = container.scrollTop;
    
    // Detect manual scroll up
    if (currentScrollTop < lastScrollTop.current) {
      isUserScrolling.current = true;
      setAutoScroll(false);
    }

    lastScrollTop.current = currentScrollTop;

    // Set a timeout to detect when user stops scrolling
    scrollTimeout.current = setTimeout(() => {
      const isAtBottom = Math.abs(
        (container.scrollHeight - container.scrollTop) - container.clientHeight
      ) < 2;

      if (isAtBottom) {
        isUserScrolling.current = false;
        setAutoScroll(true);
      }
    }, 150);
  }, []);

  // Resume auto-scrolling
  const resumeAutoScroll = useCallback(() => {
    if (tableRef.current) {
      isUserScrolling.current = false;
      setAutoScroll(true);
      requestAnimationFrame(() => {
        if (tableRef.current) {
          tableRef.current.scrollTop = tableRef.current.scrollHeight;
        }
      });
    }
  }, []);

  // Cleanup
  useEffect(() => {
    return () => {
      if (scrollTimeout.current) {
        clearTimeout(scrollTimeout.current);
      }
    };
  }, []);

  const handleRowClick = (threat: ThreatData) => {
    setSelectedThreat(threat);
  };

  const closeModal = () => {
    setSelectedThreat(null);
  };

  const handleToggleChange = () => {
    setIsAutoEmail(!isAutoEmail);
  };

  return (
    <div className="threat-table-wrapper">
      <div className="controls-container">
        {/* <button 
          onClick={sendManualEmail}
          className="send-email-button"
        >
          Send Manual Email
        </button> */}
        <div className="toggle-container">
          <ToggleSwitch isChecked={isAutoEmail} onToggle={handleToggleChange} />
          <span>{isAutoEmail ? 'Automatic Email' : 'Manual Email'}</span>
        </div>
      </div>
      {!autoScroll && (
        <button 
          className="resume-scroll-button"
          onClick={resumeAutoScroll}
        >
          Resume Auto-Scroll
        </button>
      )}
      <Modal
        isVisible={!!selectedThreat}
        onClose={closeModal}
        status={selectedThreat?.is_anomaly ? 'threat' : 'safe'}
      >
        {selectedThreat && (
          <div>
            <h2>Threat Details</h2>
            <hr />
            <p>Status: {selectedThreat.is_anomaly ? '⚠️ Threat' : '✅ Safe'}</p>
            <p>User ID: {selectedThreat.original_data.user}</p>
            <p>PC ID: {selectedThreat.original_data.pc}</p>
            <p>Date: {selectedThreat.original_data.date}</p>
            
            <p>File Operations: {[
              selectedThreat.original_data.file_open && 'Open',
              selectedThreat.original_data.file_write && 'Write',
              selectedThreat.original_data.file_copy && 'Copy',
              selectedThreat.original_data.file_delete && 'Delete',
            ].filter(Boolean).join(', ') || 'None'}</p>
            <p>Email Sent Outside: {selectedThreat.original_data.to_outside ? 'Yes' : 'No'}</p>
            <p>Email Received Outside: {selectedThreat.original_data.from_outside ? 'Yes' : 'No'}</p>
            <p>Has Attachment: {selectedThreat.original_data.has_attachment ? 'Yes' : 'No'}</p>
            <p>Is Executable: {selectedThreat.original_data.is_exe ? 'Yes' : 'No'}</p>
            <p>Logon: {selectedThreat.original_data.logon ? 'Yes' : 'No'}</p>
            <p>Auto Email: {isAutoEmail ? 'Enabled' : 'Disabled'}</p>
            {selectedThreat.is_anomaly && !isAutoEmail && (
              <button 
                onClick={() => sendEmailNotification(selectedThreat)}
                className="send-notification-button"
              >
                Send Threat Notification
              </button>
            )}
          </div>
        )}
      </Modal>
      <div 
        className="threat-table-container"
        ref={tableRef}
        onScroll={handleScroll}
      >
        <table className="threat-table">
          <thead>
            <tr>
              <th>No.</th>
              <th>Date</th>
              <th>User ID</th>
              <th>PC ID</th>
              <th>Status</th>
              <th>File Operations</th>
            </tr>
          </thead>
          <tbody>
            {data && data.map((item, index) => {
              if (!item || !item.original_data) return null;

              return (
                <tr 
                  key={`${item.original_data.date}-${index}`}
                  className={`${item.is_anomaly ? 'threat-row' : 'safe-row'} ${
                    index === data.length - 1 ? 'new-row' : ''
                  }`}
                  onClick={() => handleRowClick(item)}
                >
                  <td>{index + 1}</td>
                  <td>{item.original_data.date}</td>
                  <td>{item.original_data.user}</td>
                  <td>{item.original_data.pc}</td>
                  <td>
                    {item.is_anomaly ? '⚠️ Threat' : '✅ Safe'}
                  </td>
                  <td>
                    {[
                      item.original_data.file_open && 'Open',
                      item.original_data.file_write && 'Write',
                      item.original_data.file_copy && 'Copy',
                      item.original_data.file_delete && 'Delete'
                    ].filter(Boolean).join(', ') || 'None'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ThreatTable; 