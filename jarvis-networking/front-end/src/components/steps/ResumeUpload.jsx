import React, { useState } from 'react';

const ResumeUpload = ({ onContinue }) => {
    const [fileName, setFileName] = useState('');
    const [fileLoaded, setFileLoaded] = useState(false);
    const [base64, setBase64] = useState('');
    const [loading, setLoading] = useState(false);
  
    const handleFileChange = (e) => {
      const file = e.target.files[0];
      if (file && file.type === 'application/pdf') {
        setFileName(file.name);
        const reader = new FileReader();
        reader.onload = () => {
          setBase64(reader.result);
          setFileLoaded(true);
        };
        reader.readAsDataURL(file);
      }
    };
  
    const handleContinue = () => {
      if (fileName && base64) {
        onContinue({ name: fileName, base64 });
      }
    };
  
    return (
      <div className="text-center border border-gray-300 p-6 rounded-lg space-y-4">
        <h2 className="text-xl font-semibold">Upload Your Resume</h2>
        <input type="file" accept=".pdf,.docx" onChange={handleFileChange} />
  
        {fileLoaded && (
          <div className="flex items-center justify-between border mt-4 px-4 py-2 rounded-lg bg-gray-50">
            <div className="flex items-center gap-2">
              <span className="text-xl">📄</span>
              <div className="text-left">
                <p className="font-medium">{fileName}</p>
                <p className="text-sm text-gray-500">PDF Document • Ready for analysis</p>
              </div>
            </div>
            <button
            onClick={async () => {
            setLoading(true);
            try {
                  await onContinue({ name: fileName, base64 });
                } catch (err) {
                  console.error("Failed to parse resume:", err);
                  alert("Something went wrong parsing the resume.");
                } finally {
                  setLoading(false);
                }
              }}
              disabled={loading}
              className={`px-4 py-2 rounded text-white ${loading ? 'bg-gray-500 cursor-not-allowed' : 'bg-black'}`}
            >
              {loading ? (
                  <div className="flex items-center gap-2">
                  <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  <span>Parsing...</span>
                  </div>
                ) : (
                  'Continue'
                  )}
            </button>
          </div>
        )}
      </div>
    );
  };
  

export default ResumeUpload;
