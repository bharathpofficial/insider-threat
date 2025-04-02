const express = require('express');
const SparkPost = require('sparkpost');

const app = express();
const client = new SparkPost('8687b81c93535b72adad513c33d4e488b2b6c4dc'); // Replace with your actual API key

app.use(express.static('public'));

app.post('/send-email', async (req, res) => {
  console.log('sendManualEmail function called');
  try {
    // Define the email content
    const emailContent = `
      <h1>Manual Email Notification</h1>
      <p>This is a manually triggered email notification.</p>
    `;

    // Send the email
    const response = await client.transmissions.send({
      content: {
        from: 'security@auto-email.mycybersecurity.systems',
        subject: 'Manual Email Notification',
        html: emailContent,
      },
      recipients: [{ address: 'bharathp@securefoss.tech' }],
    });

    console.log('Email sent successfully:', response);
    res.json({ message: 'Email sent successfully' });
  } catch (error) {
    console.error('Error sending email:', error);
    res.status(500).json({ error: 'Error sending email' });
  }
});

// Serve the test email HTML page
app.get('/', (req, res) => {
  res.sendFile(__dirname + '/testEmail.html');
});


app.listen(3000, () => {
  console.log('Server is running on http://localhost:3000');
});