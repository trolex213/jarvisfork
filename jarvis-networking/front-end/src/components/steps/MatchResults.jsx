import React, { useState, useEffect, useRef } from 'react';
import { X, Check, Loader } from 'lucide-react';
import EmailGeneratorModals from '../modals/EmailGeneratorModals';

const dummyMatches = Array.from({ length: 30 }, (_, i) => ({
  name: `Candidate ${i + 1}`,
  position: "Associate",
  company: "Goldman Sachs",
  department: "TMT",
  email: `candidate${i + 1}@gs.com`,
  linkedin: `linkedin.com/in/candidate${i + 1}`,
  similarities: [
    "Both studied finance",
    "Worked in similar sectors",
    "Shared mentors or schools"
  ]
}));

const MatchResults = ({
  resumeData,
  matches = dummyMatches,
  numResults = 20,
  onBack
}) => {
  const [selectedContacts, setSelectedContacts] = useState([]);
  const [isGeneratingEmail, setIsGeneratingEmail] = useState(false);
  const [showEmailModal, setShowEmailModal] = useState(false);
  const [visibleCount, setVisibleCount] = useState(Math.min(3, Math.min(matches.length, numResults)));
  const [loadingMore, setLoadingMore] = useState(false);

  const loadMoreRef = useRef(null);
  const maxVisible = Math.min(numResults, matches.length);

  const toggleContactSelection = (index) => {
    setSelectedContacts((prev) =>
      prev.includes(index) ? prev.filter(i => i !== index) : [...prev, index]
    );
  };

  const generateEmails = () => {
    setIsGeneratingEmail(true);
    setTimeout(() => {
      setIsGeneratingEmail(false);
      setShowEmailModal(true);
    }, 1000);
  };

  const loadMore = () => {
    if (visibleCount < maxVisible && !loadingMore) {
      setLoadingMore(true);
      setTimeout(() => {
        setVisibleCount((prev) => Math.min(prev + 3, maxVisible));
        setLoadingMore(false);
      }, 800);
    }
  };

  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) loadMore();
    }, { threshold: 1 });

    const node = loadMoreRef.current;
    if (node) observer.observe(node);
    return () => node && observer.unobserve(node);
  }, [visibleCount, maxVisible]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-medium text-gray-900">Your Professional Connections</h3>
        <button onClick={onBack} className="text-sm text-gray-500 hover:text-gray-700 transition">Refine Search</button>
      </div>

      {/* Email Generator Modal */}
      {showEmailModal && (
        <EmailGeneratorModals
          selectedContacts={selectedContacts.map(i => matches[i])}
          onClose={() => setShowEmailModal(false)}
        />
      )}

      {/* Email Section */}
      <div className="bg-white border border-gray-100 rounded-2xl p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h4 className="text-base font-medium text-gray-900">Create Personalized Outreach Emails</h4>
            <p className="text-sm text-gray-500">Select contacts to generate personalized emails</p>
          </div>
          <button
            onClick={generateEmails}
            disabled={selectedContacts.length === 0 || isGeneratingEmail}
            className={`px-4 py-2 rounded-full text-sm font-medium ${
              selectedContacts.length > 0 && !isGeneratingEmail
                ? 'bg-black text-white hover:bg-gray-800'
                : 'bg-gray-100 text-gray-400'
            } transition`}
          >
            {isGeneratingEmail ? (
              <>
                <Loader className="h-4 w-4 animate-spin mr-2 inline" />
                Generating...
              </>
            ) : (
              'Generate Emails'
            )}
          </button>
        </div>

        <div className="bg-gray-50 rounded-xl p-4 flex flex-wrap gap-2 min-h-16">
          {selectedContacts.map(index => (
            <div key={index} className="bg-gray-100 text-gray-800 text-xs px-2 py-1 rounded-md flex items-center">
              {matches[index].name}
              <button onClick={() => toggleContactSelection(index)} className="ml-1 text-gray-500 hover:text-gray-700">
                <X className="h-3 w-3" />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Match Cards */}
      <div className="space-y-4">
        {matches.slice(0, visibleCount).map((match, index) => (
          <div key={index} className="border border-gray-200 rounded-2xl p-6 hover:shadow-md transition bg-white">
            <div className="flex justify-between items-start">
              <div className="flex items-start">
                <div className="h-12 w-12 bg-gray-200 rounded-full mr-4 flex items-center justify-center text-gray-600 font-medium">
                  {match.name.split(' ').map(n => n[0]).join('')}
                </div>
                <div>
                  <h4 className="font-medium text-base text-gray-900">{match.name}</h4>
                  <p className="text-gray-500">{match.position} at {match.company}, {match.department}</p>
                  <div className="flex items-center mt-1 space-x-4">
                    <a href={`https://${match.linkedin}`} target="_blank" rel="noopener noreferrer" className="text-xs text-gray-500 hover:text-black transition">LinkedIn</a>
                    <a href={`mailto:${match.email}`} className="text-xs text-gray-500 hover:text-black transition">{match.email}</a>
                  </div>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <div className="bg-black text-white text-xs font-medium px-3 py-1 rounded-full">Match #{index + 1}</div>
                <button
                  onClick={() => toggleContactSelection(index)}
                  className={`h-6 w-6 rounded-full border flex items-center justify-center transition ${
                    selectedContacts.includes(index)
                      ? 'bg-black border-black text-white'
                      : 'border-gray-300 text-transparent hover:border-gray-400'
                  }`}
                >
                  <Check className="h-3 w-3" />
                </button>
              </div>
            </div>
            <div className="mt-4 pl-16">
              <p className="text-xs font-medium uppercase tracking-wider text-gray-500 mb-2">Why you're similar</p>
              <div className="flex flex-wrap gap-2">
                {match.similarities.map((sim, i) => (
                  <span key={i} className="bg-gray-100 text-sm text-gray-800 px-3 py-1 rounded-full flex items-center">
                    <Check className="h-3 w-3 text-black mr-1" />
                    {sim}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Infinite Scroll Loader */}
      {visibleCount < maxVisible && (
        <div className="mt-8 text-center" ref={loadMoreRef}>
          <p className="text-sm text-gray-500 mb-2">
            Showing {visibleCount} of your requested {maxVisible} results
          </p>
          <Loader className="h-5 w-5 animate-spin mx-auto text-gray-400" />
        </div>
      )}
    </div>
  );
};

export default MatchResults;
