import React, { useState } from 'react';

const SearchParameters = ({ parsedResume, onSearch, onBack }) => {
  const [query, setQuery] = useState('');
  const [count, setCount] = useState(20);
  const [showJson, setShowJson] = useState(false);

  const handleSearch = () => {
    const dummyResults = [
      {
        name: 'Nicole B.',
        role: 'Analyst at Goldman Sachs, TMT',
        tags: ['Both went to Central Bucks HS', 'Both are Wharton Research Scholars']
      },
      {
        name: 'Tommy H.',
        role: 'VP at Goldman Sachs, TMT',
        tags: ['MBA at Chicago Booth', 'PIMCO internship overlap']
      },
      {
        name: 'Amy Z.',
        role: 'Analyst at Goldman Sachs, TMT',
        tags: ['Same major, same city']
      }
    ];
    onSearch(query, dummyResults);
  };

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold">Search Parameters</h2>
      <textarea
        className="w-full border p-2 rounded"
        placeholder="e.g., Looking for Goldman Sachs TMT internship..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <input
        type="number"
        className="border p-2 rounded w-24"
        value={count}
        onChange={(e) => setCount(parseInt(e.target.value))}
      />

      <div className="flex gap-4">
        <button onClick={onBack} className="px-4 py-2 bg-gray-300 rounded">Back</button>
        <button onClick={handleSearch} className="px-4 py-2 bg-black text-white rounded">Find Connections</button>
      </div>

      <div>
        <button
          onClick={() => setShowJson(!showJson)}
          className="mt-6 text-sm underline text-blue-600"
        >
          {showJson ? 'Hide Parsed Resume JSON' : 'Show Parsed Resume JSON'}
        </button>
        {showJson && (
          <pre className="mt-2 bg-gray-100 p-4 rounded overflow-x-auto text-sm max-h-[400px]">
            {JSON.stringify(parsedResume, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
};

export default SearchParameters;
