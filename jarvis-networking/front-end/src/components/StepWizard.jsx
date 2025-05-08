import React, { useState } from 'react';
import ResumeUpload from './steps/ResumeUpload';
import SearchParameters from './steps/SearchParameters';
import MatchResults from './steps/MatchResults';
import BaseLayout from './BaseLayout';

const StepWizard = () => {
  const [step, setStep] = useState(1);
  const [resumeData, setResumeData] = useState(null);
  const [parsedJson, setParsedJson] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [matches, setMatches] = useState([]);

  const goToNext = () => setStep((prev) => prev + 1);
  const goToPrev = () => setStep((prev) => prev - 1);

  const handleResumeContinue = async (data) => {
    setResumeData(data);
    setParsedJson(data.parsed);
    goToNext();
  };

  return (
    <BaseLayout>
      {/* Step Indicator */}
      <div className="flex items-center justify-center mb-12">
  <div className={`flex items-center justify-center h-10 w-10 rounded-full ${
    step >= 1 ? 'bg-black text-white' : 'bg-gray-100 text-gray-400'
  }`}>
    1
  </div>
  <div className={`h-0.5 w-16 ${step >= 2 ? 'bg-black' : 'bg-gray-200'}`}></div>
  <div className={`flex items-center justify-center h-10 w-10 rounded-full ${
    step >= 2 ? 'bg-black text-white' : 'bg-gray-100 text-gray-400'
  }`}>
    2
  </div>
  <div className={`h-0.5 w-16 ${step >= 3 ? 'bg-black' : 'bg-gray-200'}`}></div>
  <div className={`flex items-center justify-center h-10 w-10 rounded-full ${
    step >= 3 ? 'bg-black text-white' : 'bg-gray-100 text-gray-400'
  }`}>
    3
  </div>
</div>

      {/* Step Screens */}
      {step === 1 && (
        <ResumeUpload onContinue={handleResumeContinue} />
      )}
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
    </BaseLayout>
  );
};

export default StepWizard;