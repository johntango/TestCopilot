const express = require('express');
const app = express();
const path = require('path');

// Define the directory where the image is stored
const imagePath = path.join(__dirname, 'images');

// Set up a route for the default route ("/")
app.get('/', (req, res) => {
  const imagePath = path.join(__dirname, 'images', 'face.jpg');
  res.sendFile(imagePath);
});

// Start the server
app.listen(3000, () => {
  console.log('Server started on port 3000');
});