import React, { useState } from 'react';

const MatchResults = ({ resumeData, matches, onBack }) => {
  const [selected, setSelected] = useState([]);
  const [sample, setSample] = useState('');

  const generateEmails = () => {
    alert(`Generated emails for: ${selected.join(', ')}`);
  };

  const toggleSelect = (name) => {
    setSelected((prev) =>
      prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name]
    );
  };

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold">Your Professional Connections</h2>
      {matches.map((match, i) => (
        <div key={i} className="border p-4 rounded shadow-sm">
          <div className="flex justify-between items-center">
            <div>
              <h3 className="font-bold">{match.name}</h3>
              <p>{match.role}</p>
            </div>
            <label>
              <input
                type="checkbox"
                checked={selected.includes(match.name)}
                onChange={() => toggleSelect(match.name)}
              /> Match #{i + 1}
            </label>
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {match.tags.map((tag, idx) => (
              <span key={idx} className="px-2 py-1 bg-gray-100 text-sm rounded">
                ✓ {tag}
              </span>
            ))}
          </div>
        </div>
      ))}

      <div className="space-y-2">
        <textarea
          className="w-full border p-2 rounded"
          placeholder="Paste your email tone sample here..."
          value={sample}
          onChange={(e) => setSample(e.target.value)}
        />
        <button
          className="px-4 py-2 bg-black text-white rounded"
          onClick={generateEmails}
        >
          Generate Emails
        </button>
      </div>

      <button onClick={onBack} className="px-4 py-2 bg-gray-300 rounded">Back</button>
    </div>
  );
};

export default MatchResults;