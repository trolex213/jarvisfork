import React, { useState, useEffect } from 'react';
import ResumeUpload from './steps/ResumeUpload';
import SearchParameters from './steps/SearchParameters';
import MatchResults from './steps/MatchResults';

const StepWizard = () => {
  const [step, setStep] = useState(1);
  const [resumeData, setResumeData] = useState(null);
  const [parsedJson, setParsedJson] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [matches, setMatches] = useState([]);

  const goToNext = () => setStep((prev) => prev + 1);
  const goToPrev = () => setStep((prev) => prev - 1);

  const handleResumeContinue = async (data) => {
    if (!parsedJson) {
      const response = await fetch('http://127.0.0.1:8000/parse_resume', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ base64: data.base64 })
      });

      const result = await response.json();
      console.log(result);
      setParsedJson(result);
    }
    setResumeData(data);
    goToNext();
  };

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <div className="flex justify-center gap-4 mb-6">
        {[1, 2, 3].map((s) => (
          <div
            key={s}
            className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-white ${
              step === s ? 'bg-black' : 'bg-gray-300'
            }`}
          >
            {s}
          </div>
        ))}
      </div>

      {step === 1 && <ResumeUpload onContinue={handleResumeContinue} />}

      {step === 2 && (
        <SearchParameters
          parsedResume={parsedJson}
          onSearch={(query, results) => {
            setSearchQuery(query);
            setMatches(results);
            goToNext();
          }}
          onBack={goToPrev}
        />
      )}

      {step === 3 && (
        <MatchResults
          resumeData={resumeData}
          matches={matches}
          onBack={goToPrev}
        />
      )}
    </div>
  );
};

export default StepWizard;