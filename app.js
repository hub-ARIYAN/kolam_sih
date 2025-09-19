const express = require('express');
const multer = require('multer');
const path = require('path');
const { exec } = require('child_process');
const fs = require('fs');

const app = express();
const PORT = 3000;

// Set EJS as templating engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

// Multer config for uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, path.join(__dirname, 'public', 'uploads')),
  filename: (req, file, cb) => cb(null, Date.now() + path.extname(file.originalname))
});
const upload = multer({ storage });

// Home page
app.get('/', (req, res) => {
  res.render('index');
});

// Handle upload and analysis
app.post('/analyze', upload.single('kolamImage'), (req, res) => {
  if (!req.file) return res.status(400).send('No file uploaded.');

  const imgPath = path.join(__dirname, 'public', 'uploads', req.file.filename);
  console.log("3jfnkjfenfwknweke");
  exec(`python Kolam_Tester.py "${imgPath}" "public/results"`, { timeout: 120000 }, (err, stdout, stderr) => {
    console.log('Kolam_Tester.py stdout:', stdout);
    console.error('Kolam_Tester.py stderr:', stderr);
    if (err) {
      return res.status(500).send(`<pre>Kolam_Tester.py failed:\n${stderr || err.message}</pre>`);
    }

    exec(`python main_eq_conv.py "${imgPath}" "public/results"`, { timeout: 120000 }, (err2, stdout2, stderr2) => {
      console.log('main_eq_conv.py stdout:', stdout2);
      console.error('main_eq_conv.py stderr:', stderr2);
      if (err2) {
        return res.status(500).send(`<pre>main_eq_conv.py failed:\n${stderr2 || err2.message}</pre>`);
      }

      // Find the latest *_analysis_visualization.png in public/results
      const resultsDir = path.join(__dirname, 'public', 'results');
      let vizFile = null;
      const files = fs.readdirSync(resultsDir)
        .filter(f => f.endsWith('_analysis_visualization.png'))
        .map(f => ({ file: f, time: fs.statSync(path.join(resultsDir, f)).mtime.getTime() }))
        .sort((a, b) => b.time - a.time);
      if (files.length > 0) {
        vizFile = '/results/' + files[0].file;
      } else {
        vizFile = null;
      }

      // Render result page
      res.render('result', {
        image: vizFile,
        analysisFile: '/results/analysis.txt',
        equationsFile: '/results/equations.txt',
        desmosFile: '/results/output_eq.html'
      });
    });
  });
});

// Download endpoints
app.get('/download/:file', (req, res) => {
  const file = req.params.file;
  const filePath = path.join(__dirname, 'public', 'results', file);
  if (fs.existsSync(filePath)) {
    res.download(filePath);
  } else {
    res.status(404).send('File not found');
  }
});

// View Desmos (spiderman.html)
app.get('/view-desmos', (req, res) => {
  const filePath = path.join(__dirname, 'public', 'results', 'output_eq.html');
  if (fs.existsSync(filePath)) {
    res.sendFile(filePath);
  } else {
    res.status(404).send('Desmos visualization not found');
  }
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});