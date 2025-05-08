import React, { useState } from 'react';
import { Loader } from 'lucide-react';

const SearchParameters = ({ parsedResume, onSearch, onBack }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [numResults, setNumResults] = useState(20);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = () => {
    if (!searchQuery.trim()) {
      setError('Please specify search criteria');
      return;
    }

    setError('');
    setIsLoading(true);

    // Simulate dummy matches
    setTimeout(() => {
      const dummyResults = [
        {
          name: 'Nicole B******',
          position: 'Analyst',
          company: 'Goldman Sachs',
          department: 'TMT',
          email: 'nicole.b@gs.com',
          linkedin: 'linkedin.com/in/nicole-b',
          similarities: [
            'Both went to Central Bucks HS',
            'Both are Wharton Research Scholars'
          ]
        },
        {
          name: 'Tommy H******',
          position: 'VP',
          company: 'Goldman Sachs',
          department: 'TMT',
          email: 'tommy.h@gs.com',
          linkedin: 'linkedin.com/in/tommy-h',
          similarities: [
            'MBA at Chicago Booth',
            'PIMCO internship overlap'
          ]
        }
      ];

      onSearch(searchQuery, dummyResults);
      setIsLoading(false);
    }, 1500);
  };

  return (
    <div className="mb-12">
      <h3 className="text-xl font-medium text-gray-900 mb-6">Search Parameters</h3>

      <div className="border border-gray-200 rounded-2xl p-8 bg-gray-50 shadow-sm">
        <div className="mb-6">
          <label htmlFor="search-query" className="block text-sm font-medium text-gray-700 mb-2">
            What kind of professionals are you looking for?
          </label>
          <input
            id="search-query"
            type="text"
            placeholder="E.g., 20 people from Goldman Sachs TMT"
            className="w-full p-4 border border-gray-200 rounded-xl bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-black focus:border-transparent transition"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <p className="text-xs text-gray-500 mt-2">Specify company, department, or other criteria</p>
        </div>

        <div className="mb-8">
          <label htmlFor="num-results" className="block text-sm font-medium text-gray-700 mb-2">
            Number of results
          </label>
          <input
            id="num-results"
            type="number"
            min="1"
            max="50"
            className="w-32 p-4 border border-gray-200 rounded-xl bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-black focus:border-transparent transition"
            value={numResults}
            onChange={(e) => setNumResults(parseInt(e.target.value))}
          />
        </div>

        {error && (
          <div className="bg-red-50 border border-red-100 rounded-xl p-4 mb-6">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        <div className="flex items-center justify-between">
          <button
            onClick={onBack}
            className="text-gray-500 font-medium hover:text-gray-700 transition"
          >
            Back
          </button>

          <button
            onClick={handleSubmit}
            className="bg-black text-white px-6 py-3 rounded-full flex items-center justify-center font-medium transition hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <Loader className="h-4 w-4 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              'Find Connections'
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default SearchParameters;
