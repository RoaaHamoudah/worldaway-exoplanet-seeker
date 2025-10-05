import React, { useState, useEffect, useCallback, memo } from 'react';
import { Upload, Sparkles, AlertCircle, Download, TrendingUp, Database, Eye } from 'lucide-react';
import SplineBackground from './SplineBackground';



// Glitch Text Effect
const GlitchText = ({ children, className = '' }) => {
  return (
    <div className={`relative ${className}`}>
      <span className="relative z-10">{children}</span>
      <span className="absolute top-0 left-0 text-red-500 opacity-70 animate-glitch-1" aria-hidden="true">
        {children}
      </span>
      <span className="absolute top-0 left-0 text-cyan-400 opacity-70 animate-glitch-2" aria-hidden="true">
        {children}
      </span>
    </div>
  );
};

function App() {
  const [currentPage, setCurrentPage] = useState('landing');
  const [file, setFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [jobResults, setJobResults] = useState(null);
  const [formData, setFormData] = useState({
    koi_period: '',
    koi_time0bk: '',
    koi_impact: '',
    koi_duration: '',
    koi_depth: '',
    koi_prad: '',
    koi_teq: '',
    koi_insol: '',
    koi_model_snr: '',
    koi_steff: '',
    koi_slogg: '',
    koi_srad: '',
    koi_kepmag: ''
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [soundEnabled, setSoundEnabled] = useState(false);

  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  // poll job status
  useEffect(() => {
    if (jobId && jobStatus?.status === 'processing') {
      const interval = setInterval(async () => {
        try {
          const response = await fetch(`${API_URL}/api/job/${jobId}/status`);
          const data = await response.json();
          setJobStatus(data);
          
          if (data.status === 'completed' || data.status === 'failed') {
            clearInterval(interval);
            if (data.status === 'completed') {
              fetchJobResults();
            }
          }
        } catch (err) {
          console.error('Error polling status:', err);
        }
      }, 2000);

      return () => clearInterval(interval);
    }
  }, [jobId, jobStatus, API_URL]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleFileUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_URL}/api/predict/file`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();
      setJobId(data.job_id);
      setJobStatus({ status: 'processing', progress: 0 });
      setCurrentPage('results');
    } catch (err) {
      setError('Failed to upload file. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleFormChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleFormSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    const requestData = {};
    Object.keys(formData).forEach(key => {
      requestData[key] = parseFloat(formData[key]) || 0;
    });

    try {
      const response = await fetch(`${API_URL}/api/predict/row`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      setPrediction(data);
      
      // play sound if confirmed and sound enabled
      if (data.predicted_label === 'CONFIRMED' && soundEnabled) {
        const audio = new Audio('/sounds/confirmed.mp3');
        audio.play().catch(e => console.log('Audio play failed:', e));
      }
      
      setCurrentPage('predict-result');
    } catch (err) {
      setError('Failed to get prediction. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const downloadResults = async () => {
    if (!jobId) return;

    try {
      const response = await fetch(`${API_URL}/api/job/${jobId}/download`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `exoplanet_predictions_${jobId}.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError('Failed to download results');
      console.error(err);
    }
  };

  const fetchJobResults = async () => {
    if (!jobId) return;

    try {
      const response = await fetch(`${API_URL}/api/job/${jobId}/results?page=1&page_size=10`);
      const data = await response.json();
      setJobResults(data);
    } catch (err) {
      console.error('Error fetching results:', err);
    }
  };

  // landing Page
  const LandingPage = () => (
    <div className="min-h-screen flex flex-col items-center justify-center p-8 text-center">
      <h1 className="text-6xl font-bold mb-4 text-white">
        WorldAway
      </h1>
      <p className="text-2xl mb-8 text-[#9BE3FF] font-medium">Exoplanet Seeker</p>
      
      <p className="text-lg mb-12 text-gray-300 max-w-2xl italic leading-relaxed">
        "In the void between stars, we search for worlds that might harbor life... 
        or horrors beyond comprehension."
      </p>
      
      <div className="flex gap-6">
        <button
          onClick={() => setCurrentPage('upload')}
          className="flex items-center gap-2 bg-[#E23B3B] hover:bg-red-600 text-white px-8 py-4 rounded-lg text-lg font-semibold transition-all transform hover:scale-105 shadow-lg shadow-red-900/50"
        >
          <Upload size={24} />
          Upload Data
        </button>
        
        <button
          onClick={() => setCurrentPage('predict')}
          className="flex items-center gap-2 bg-[#9BE3FF] hover:bg-blue-400 text-[#0B1221] px-8 py-4 rounded-lg text-lg font-semibold transition-all transform hover:scale-105 shadow-lg shadow-blue-500/50"
        >
          <Sparkles size={24} />
          Predict Now
        </button>
      </div>

      <div className="mt-12 flex items-center gap-3 text-gray-400">
        <input
          type="checkbox"
          id="sound"
          checked={soundEnabled}
          onChange={(e) => setSoundEnabled(e.target.checked)}
          className="w-5 h-5"
        />
        <label htmlFor="sound" className="text-sm">
          Enable confirmation sound effect
        </label>
      </div>
    </div>
  );

  // upload page
  const UploadPage = () => (
    <div className="min-h-screen flex flex-col items-center justify-center p-8">
      <div className="bg-[#0B1221]/80 backdrop-blur-lg border-2 border-[#9BE3FF]/30 rounded-xl p-8 max-w-2xl w-full shadow-2xl">
        <h2 className="text-4xl font-bold mb-6 text-[#9BE3FF]">
          <Database className="inline mr-3" />
          Upload Exoplanet Data
        </h2>
        
        <div className="mb-6 p-4 bg-[#E23B3B]/10 border border-[#E23B3B]/30 rounded-lg">
          <p className="text-sm text-gray-300">
            <AlertCircle className="inline mr-2" size={16} />
            Required columns: koi_period, koi_time0bk, koi_impact, koi_duration, koi_depth, 
            koi_prad, koi_teq, koi_insol, koi_model_snr, koi_steff, koi_slogg, koi_srad, koi_kepmag
          </p>
        </div>

        <div className="border-2 border-dashed border-[#9BE3FF]/50 rounded-lg p-12 text-center mb-6 hover:border-[#9BE3FF] transition-colors">
          <input
            type="file"
            accept=".xlsx,.xls,.csv"
            onChange={handleFileChange}
            className="hidden"
            id="file-upload"
          />
          <label htmlFor="file-upload" className="cursor-pointer">
            <Upload size={48} className="mx-auto mb-4 text-[#9BE3FF]" />
            <p className="text-lg text-gray-300">
              {file ? file.name : 'Click to select Excel or CSV file'}
            </p>
            <p className="text-sm text-gray-500 mt-2">Max 200MB, up to 10,000 rows</p>
          </label>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-[#E23B3B]/20 border border-[#E23B3B] rounded-lg text-white">
            {error}
          </div>
        )}

        <div className="flex gap-4">
          <button
            onClick={() => setCurrentPage('landing')}
            className="flex-1 bg-gray-700 hover:bg-gray-600 text-white px-6 py-3 rounded-lg transition-colors"
          >
            Back
          </button>
          
          <button
            onClick={handleFileUpload}
            disabled={!file || loading}
            className="flex-1 bg-[#E23B3B] hover:bg-red-600 text-white px-6 py-3 rounded-lg font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Uploading...' : 'Process File'}
          </button>
        </div>
      </div>
    </div>
  );

  // manual predict page
  const PredictPage = () => (
    <div className="min-h-screen flex flex-col items-center justify-center p-8">
      <div className="bg-[#0B1221]/80 backdrop-blur-lg border-2 border-[#9BE3FF]/30 rounded-xl p-8 max-w-4xl w-full shadow-2xl">
        <h2 className="text-4xl font-bold mb-6 text-[#9BE3FF]">
          <Sparkles className="inline mr-3" />
          Manual Prediction
        </h2>

        <form onSubmit={handleFormSubmit} className="grid grid-cols-2 gap-4">
          {Object.keys(formData).map((key) => (
            <div key={key}>
              <label className="block text-sm text-gray-300 mb-2 capitalize">
                {key.replace('koi_', '').replace('_', ' ')}
              </label>
              <input
                type="text"
                name={key}
                value={formData[key]}
                onChange={handleFormChange}
                className="w-full bg-[#0B1221] border border-[#9BE3FF]/30 rounded px-4 py-2 text-white focus:outline-none focus:border-[#9BE3FF]"
                placeholder="Enter value"
                autoComplete="off"
              />
            </div>
          ))}

          {error && (
            <div className="col-span-2 p-4 bg-[#E23B3B]/20 border border-[#E23B3B] rounded-lg text-white">
              {error}
            </div>
          )}

          <div className="col-span-2 flex gap-4 mt-4">
            <button
              type="button"
              onClick={() => setCurrentPage('landing')}
              className="flex-1 bg-gray-700 hover:bg-gray-600 text-white px-6 py-3 rounded-lg transition-colors"
            >
              Back
            </button>
            
            <button
              type="submit"
              disabled={loading}
              className="flex-1 bg-[#E23B3B] hover:bg-red-600 text-white px-6 py-3 rounded-lg font-semibold transition-colors disabled:opacity-50"
            >
              {loading ? 'Predicting...' : 'Predict'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );

  // results page (for file upload)
  const ResultsPage = () => (
    <div className="min-h-screen flex flex-col items-center justify-center p-8">
      <div className="bg-[#0B1221]/80 backdrop-blur-lg border-2 border-[#9BE3FF]/30 rounded-xl p-8 max-w-2xl w-full shadow-2xl">
        <h2 className="text-4xl font-bold mb-6 text-[#9BE3FF]">
          <TrendingUp className="inline mr-3" />
          Processing Results
        </h2>

        {jobStatus && (
          <div className="space-y-6">
            <div>
              <div className="flex justify-between mb-2 text-gray-300">
                <span>Status: {jobStatus.status.toUpperCase()}</span>
                <span>{jobStatus.progress?.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-4 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-[#E23B3B] to-[#9BE3FF] transition-all duration-500"
                  style={{ width: `${jobStatus.progress || 0}%` }}
                />
              </div>
            </div>

            {jobStatus.total_rows && (
              <p className="text-gray-300">
                Processed {jobStatus.processed_rows || 0} of {jobStatus.total_rows} rows
              </p>
            )}

            {jobStatus.status === 'completed' && (
              <div className="space-y-4">
                <button
                  onClick={downloadResults}
                  className="w-full bg-[#E23B3B] hover:bg-red-600 text-white px-6 py-3 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2"
                >
                  <Download size={20} />
                  Download Results (CSV)
                </button>
                
                {jobResults && (
                  <div className="mt-6">
                    <h3 className="text-xl font-semibold mb-4 text-[#9BE3FF]">
                      Sample Results (First 10 rows)
                    </h3>
                    <div className="bg-[#0B1221]/50 rounded-lg p-4 max-h-96 overflow-y-auto">
                      <div className="grid grid-cols-1 gap-2 text-sm">
                        {jobResults.results.map((row, index) => (
                          <div key={index} className="flex justify-between items-center p-2 bg-[#0B1221]/30 rounded">
                            <div className="flex-1">
                              <span className="text-white">Row {row.row_id}</span>
                            </div>
                            <div className="text-right">
                              <span className={`px-2 py-1 rounded text-xs font-semibold ${
                                row.predicted_label === 'CONFIRMED' ? 'bg-green-900/50 text-green-300' :
                                row.predicted_label === 'CANDIDATE' ? 'bg-yellow-900/50 text-yellow-300' :
                                'bg-red-900/50 text-red-300'
                              }`}>
                                {row.predicted_label}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                      {jobResults.total_rows > 10 && (
                        <p className="text-gray-400 text-sm mt-4 text-center">
                          Showing first 10 of {jobResults.total_rows} total results
                        </p>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}

            {jobStatus.status === 'failed' && (
              <div className="p-4 bg-[#E23B3B]/20 border border-[#E23B3B] rounded-lg text-white">
                Processing failed: {jobStatus.error_message}
              </div>
            )}
          </div>
        )}

        <button
          onClick={() => setCurrentPage('landing')}
          className="w-full mt-6 bg-gray-700 hover:bg-gray-600 text-white px-6 py-3 rounded-lg transition-colors"
        >
          Back to Home
        </button>
      </div>
    </div>
  );

  // prediction result page (for manual entry)
  const PredictResultPage = () => (
    <div className="min-h-screen flex flex-col items-center justify-center p-8">
      <div className="bg-[#0B1221]/80 backdrop-blur-lg border-2 border-[#9BE3FF]/30 rounded-xl p-8 max-w-2xl w-full shadow-2xl">
        <h2 className="text-4xl font-bold mb-6 text-[#9BE3FF]">
          <Eye className="inline mr-3" />
          Prediction Result
        </h2>

        {prediction && (
          <div className="space-y-6">
            <div className={`p-6 rounded-lg border-2 ${
              prediction.predicted_label === 'CONFIRMED' ? 'bg-green-900/20 border-green-500' :
              prediction.predicted_label === 'CANDIDATE' ? 'bg-yellow-900/20 border-yellow-500' :
              'bg-red-900/20 border-red-500'
            }`}>
              <p className="text-sm text-gray-300 mb-2">Classification</p>
              <p className="text-3xl font-bold text-white">{prediction.predicted_label}</p>
            </div>

            <div>
              <p className="text-lg font-semibold mb-3 text-gray-300">Probabilities</p>
              {Object.entries(prediction.predicted_probs).map(([label, prob]) => (
                <div key={label} className="mb-3">
                  <div className="flex justify-between mb-1 text-gray-300">
                    <span>{label}</span>
                    <span>{(prob * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-3">
                    <div
                      className="h-full rounded-full bg-[#9BE3FF]"
                      style={{ width: `${prob * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>

            <div>
              <p className="text-lg font-semibold mb-3 text-gray-300">Top Contributing Features</p>
              {prediction.top_features.map(([feature, importance], idx) => (
                <div key={idx} className="flex justify-between p-3 bg-[#0B1221]/50 rounded mb-2">
                  <span className="text-gray-300">{feature}</span>
                  <span className="text-[#9BE3FF] font-semibold">{importance.toFixed(3)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        <button
          onClick={() => {
            setCurrentPage('predict');
            setPrediction(null);
            setFormData({
              koi_period: '', koi_time0bk: '', koi_impact: '', koi_duration: '',
              koi_depth: '', koi_prad: '', koi_teq: '', koi_insol: '',
              koi_model_snr: '', koi_steff: '', koi_slogg: '', koi_srad: '', koi_kepmag: ''
            });
          }}
          className="w-full mt-6 bg-gray-700 hover:bg-gray-600 text-white px-6 py-3 rounded-lg transition-colors"
        >
          New Prediction
        </button>
      </div>
    </div>
  );

  return (
    <div className="relative min-h-screen bg-[#0B1221] text-white overflow-hidden">
      <SplineBackground />

      {/* NASA Logo - top left */}
<div className="fixed top-4 left-4 z-20">
  <div className="flex items-center gap-3 bg-black/40 backdrop-blur-md px-4 py-2 rounded-lg border border-white/10">
    <img 
      src="https://www.nasa.gov/wp-content/themes/nasa/assets/images/nasa-logo.svg" 
      alt="NASA Logo" 
      className="w-12 h-12"
    />
    <div>
      <p className="text-white font-bold text-lg">NASA</p>
      <p className="text-gray-400 text-xs">Kepler Mission</p>
    </div>
  </div>
</div>

{/* Astro Coders Logo - top right */}
<div className="fixed top-4 right-4 z-20">
  <div className="flex items-center gap-3 bg-black/40 backdrop-blur-md px-4 py-2 rounded-lg border border-white/10">
    <div className="text-right">
      <p className="text-white font-bold text-lg">Astro Coders</p>
      <p className="text-gray-400 text-xs">Team Project</p>
    </div>
    <img 
 src="/icon.png"
  alt="Astro Coders Logo"
  className="w-12 h-12"
    />
  </div>
</div>
      
      <div className="relative z-10">
        {currentPage === 'landing' && <LandingPage />}
        {currentPage === 'upload' && <UploadPage />}
        {currentPage === 'predict' && <PredictPage />}
        {currentPage === 'results' && <ResultsPage />}
        {currentPage === 'predict-result' && <PredictResultPage />}
      </div>

    </div>
  );
}

export default App;

