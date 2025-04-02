// Remove SparkPost import and initialization
import { ThreatData } from '../types/types';

const API_BASE_URL = 'http://localhost:9090'; // Match your backend URL

export const sendEmailNotification = async (threat: ThreatData) => {
  try {
    console.log('Sending email notification for threat:', threat);
    
    const response = await fetch(`${API_BASE_URL}/api/send-email`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ threat }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error('Server error:', errorData);
      throw new Error('Failed to send email');
    }

    const data = await response.json();
    console.log('Email sent successfully:', data);
  } catch (error) {
    console.error('Error sending email:', error);
  }
};

export const sendManualEmail = async () => {
  console.log('sendManualEmail function called');
  try {
    const response = await fetch(`${API_BASE_URL}/api/send-manual-email`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error('Server error:', errorData);
      throw new Error('Failed to send manual email');
    }

    const data = await response.json();
    console.log('Manual email sent successfully:', data);
  } catch (error) {
    console.error('Error sending manual email:', error);
  }
};