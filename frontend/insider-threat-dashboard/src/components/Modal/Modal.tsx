import React, { useEffect } from 'react';
import './Modal.css';

interface ModalProps {
  isVisible: boolean;
  onClose: () => void;
  status: 'safe' | 'threat';
  children: React.ReactNode;
}

const Modal: React.FC<ModalProps> = ({ isVisible, onClose, status, children }) => {
  useEffect(() => {
    const handleEsc = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [onClose]);

  if (!isVisible) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className={`modal-content ${status === 'threat' ? 'modal-threat' : 'modal-safe'}`}
        onClick={(e) => e.stopPropagation()}
      >
        {children}
      </div>
    </div>
  );
};

export default Modal;
