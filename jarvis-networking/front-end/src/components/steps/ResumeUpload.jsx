import React, { useState } from 'react';
import { FileText, X } from 'lucide-react';

const ResumeUpload = ({ onContinue }) => {
  const [file, setFile] = useState(null);
  const [base64, setBase64] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected && selected.type === 'application/pdf') {
      setFile(selected);
      const reader = new FileReader();
      reader.onload = () => {
        setBase64(reader.result);
      };
      reader.readAsDataURL(selected);
    } else {
      alert('Please upload a valid PDF file.');
    }
  };

  const handleContinue = async () => {
    if (!file || !base64) return;
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/parse_resume', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ base64 })
      });
      const result = await response.json();
      await onContinue({ name: file.name, base64, parsed: result.parsed });
    } catch (err) {
      //console.error('❌ Failed to parse resume:', err);
      //alert('Something went wrong parsing the resume.');
      const dummyParsed = {
        name: "John Doe",
        contact: {
          email: "john@example.com",
          phone: "(123) 456-7890",
          location: "New York, NY"
        },
        education: {
          "University of Somewhere": {
            degree: "B.Sc. in Computer Science",
            date: "2018 - 2022",
            GPA: "3.9",
            coursework: "Algorithms, Databases, AI"
          }
        },
        experience: [
          {
            company: "TechCorp",
            title: "Software Engineer",
            date: "2022 - Present",
            responsibilities: "- Built full-stack apps\n- Integrated APIs\n- Led projects"
          }
        ],
        skills: ["JavaScript", "React", "Node.js", "Python"],
        certifications: ["AWS Certified Developer"],
        languages: ["English", "Spanish"],
        interests: ["Chess", "Hiking", "Open Source"]
      };
      
      await onContinue({ name: file.name, base64, parsed: dummyParsed });
      
    } finally {
      setLoading(false);
    }
  };

  const handleRemoveFile = () => {
    setFile(null);
    setBase64('');
  };

  return (
    <div className="mb-12">
      <h3 className="text-xl font-medium text-gray-900 mb-6">Upload Your Resume</h3>

      {!file ? (
        <div className="border border-gray-200 rounded-2xl p-12 text-center bg-gray-50 transition shadow-sm hover:shadow-md">
          <div className="flex flex-col items-center">
            <div className="h-16 w-16 bg-black bg-opacity-5 rounded-full flex items-center justify-center mb-6">
              <FileText className="h-8 w-8 text-black" />
            </div>
            <p className="text-gray-600 mb-6 max-w-sm mx-auto">Drag and drop your resume or click to browse files (PDF format)</p>
            <label className="bg-black text-white px-6 py-3 rounded-full cursor-pointer font-medium transition hover:bg-gray-800">
              Select Resume
              <input
                type="file"
                className="hidden"
                accept=".pdf"
                onChange={handleFileChange}
              />
            </label>
          </div>
        </div>
      ) : (
        <div className="border border-gray-200 rounded-2xl p-8 bg-gray-50 flex items-center justify-between transition shadow-sm">
          <div className="flex items-center">
            <div className="h-12 w-12 bg-black bg-opacity-5 rounded-full flex items-center justify-center mr-4">
              <FileText className="h-6 w-6 text-black" />
            </div>
            <div>
              <p className="font-medium text-gray-900">{file.name}</p>
              <p className="text-sm text-gray-500">PDF Document • Ready for analysis</p>
            </div>
          </div>
          <div className="flex items-center">
            <button
              onClick={handleRemoveFile}
              className="text-gray-400 hover:text-gray-600 mr-4"
            >
              <X className="h-5 w-5" />
            </button>
            <button
              onClick={handleContinue}
              disabled={loading}
              className="bg-black text-white px-5 py-2 rounded-full text-sm font-medium transition hover:bg-gray-800"
            >
              {loading ? 'Parsing...' : 'Continue'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResumeUpload;
