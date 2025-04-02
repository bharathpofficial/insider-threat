const express = require('express');
const SparkPost = require('sparkpost'); //  npm install spakpost
const cors = require('cors');
const fs = require('fs').promises;
const path = require('path');

const app = express();
const client = new SparkPost('8687b81c93535b72adad513c33d4e488b2b6c4dc');

// Middleware
app.use(cors()); // Enable CORS for all routes
app.use(express.json()); // Parse JSON bodies

// Function to read and populate email template
async function getEmailTemplate(templateData = null) {
  try {
    const templatePath = path.join(__dirname, 'email-template.html');
    let template = await fs.readFile(templatePath, 'utf8');
    
    if (templateData) {
      // Replace placeholders with actual data
      template = template.replace('{{threat_date}}', templateData.date || '');
      template = template.replace('{{user_id}}', templateData.user || '');
      template = template.replace('{{pc_id}}', templateData.pc || '');
      template = template.replace('{{to_outside}}', templateData.to_outside ? 'Yes' : 'No');
      template = template.replace('{{from_outside}}', templateData.from_outside ? 'Yes' : 'No');
      template = template.replace('{{has_attachment}}', templateData.has_attachment ? 'Yes' : 'No');
      template = template.replace('{{is_exe}}', templateData.is_exe ? 'Yes' : 'No');
      template = template.replace('{{file_open}}', templateData.file_open ? 'Yes' : 'No');
      template = template.replace('{{logon}}', templateData.logon ? 'Yes' : 'No');
    }
    
    return template;
  } catch (error) {
    console.error('Error reading email template:', error);
    throw error;
  }
}

// Route for sending manual emails
app.post('/api/send-manual-email', async (req, res) => {
    console.log("/send-manual-email")
  try {
    const emailContent = await getEmailTemplate();

    const response = await client.transmissions.send({
      content: {
        from: 'security@auto-email.mycybersecurity.systems',
        subject: 'Manual Email Notification',
        html: emailContent,
      },
      recipients: [{ address: 'bharathp@securefoss.tech' }],
    });

    console.log('Manual email sent successfully:', response);
    res.json({ success: true, message: 'Email sent successfully' });
  } catch (error) {
    console.error('Error sending manual email:', error);
    res.status(500).json({ success: false, error: 'Failed to send email' });
  }
});

// Route for sending threat notification emails
app.post('/api/send-email', async (req, res) => {
//   console.log('Received threat notification request:', req.body);
  
  try {
    const { threat } = req.body;
    
    if (!threat || !threat.original_data) {
      console.error('Invalid threat data received:', req.body);
      return res.status(400).json({ 
        success: false, 
        error: 'Invalid threat data' 
      });
    }

    // console.log('Processing threat data:', threat.original_data);
    const emailContent = await getEmailTemplate(threat.original_data);

    const response = await client.transmissions.send({
      content: {
        from: 'security@auto-email.mycybersecurity.systems',
        subject: 'Threat Alert Notification',
        html: emailContent,
      },
      recipients: [{ address: 'bharathp@securefoss.tech' }],
    });

    console.log('Threat notification email sent successfully:', response);
    res.json({ success: true, message: 'Email sent successfully' });
  } catch (error) {
    console.error('Error sending threat notification:', error);
    res.status(500).json({ 
      success: false, 
      error: 'Failed to send email',
      details: error.message 
    });
  }
});

// Start the server
const PORT = 9090;
const server = app.listen(PORT, () => {
    console.log(`Email server running on http://localhost:${PORT}`);
}).on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.log(`Port ${PORT} is already in use. Trying port ${PORT + 1}`);
    server.listen(PORT + 1);
    console.log(`Email server running on http://localhost:${PORT}`);
  } else {
    console.error('Server error:', err);
  }
});