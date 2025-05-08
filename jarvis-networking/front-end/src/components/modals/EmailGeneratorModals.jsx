import React, { useState } from 'react';
import { X } from 'lucide-react';

const EmailGeneratorModals = ({
  selectedContacts,
  onClose,
  copyToClipboard
}) => {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const contact = selectedContacts[selectedIndex];

  const subject = `Let's Connect – Shared Finance Background`;
  const emailBody = `Dear ${contact.name},

My name is Laura, and I'm a rising junior at Wharton studying Finance. I'm currently in the Wharton Research Scholars program.

I was excited to come across your profile at ${contact.company} in the ${contact.department} division. It’s always inspiring to see someone who's pursued a similar path.

I'm currently exploring internship opportunities and would be grateful to connect and learn about your experience. Even a short chat would be deeply appreciated.

Thank you so much — I look forward to hearing from you!

Warmly,
Laura`;

  const goPrev = () => setSelectedIndex((i) => Math.max(i - 1, 0));
  const goNext = () => setSelectedIndex((i) => Math.min(i + 1, selectedContacts.length - 1));

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl w-full max-w-3xl max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-gray-100 flex justify-between items-center">
          <h3 className="text-xl font-medium text-gray-900">Personalized Outreach Emails</h3>
          <button
            onClick= {onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto flex-1 space-y-6">
          {/* Navigation */}
          <div className="flex items-center justify-between mb-2">
            <button
              onClick={goPrev}
              disabled={selectedIndex === 0}
              className="bg-gray-100 hover:bg-gray-200 text-gray-700 p-2 rounded-full transition disabled:opacity-30"
            >
              ←
            </button>
            <span className="text-sm text-gray-500">
              {selectedIndex + 1} of {selectedContacts.length} emails
            </span>
            <button
              onClick={goNext}
              disabled={selectedIndex === selectedContacts.length - 1}
              className="bg-gray-100 hover:bg-gray-200 text-gray-700 p-2 rounded-full transition disabled:opacity-30"
            >
              →
            </button>
          </div>

          {/* Email Card */}
          <div className="border border-gray-200 rounded-xl p-4">
            <div className="flex items-center mb-4">
              <div className="h-10 w-10 bg-gray-200 rounded-full mr-3 flex items-center justify-center text-gray-500 font-medium">
                {contact.name.split(' ').map(n => n[0]).join('')}
              </div>
              <div>
                <h4 className="font-medium text-base text-gray-900">{contact.name}</h4>
                <p className="text-xs text-gray-500">{contact.email}</p>
              </div>
            </div>

            <div className="border-t border-gray-100 pt-4">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-medium text-gray-900">Subject:</p>
                <button
                  onClick={() => copyToClipboard(subject)}
                  className="text-xs text-gray-500 hover:text-black transition"
                >
                  Copy
                </button>
              </div>
              <p className="text-sm text-gray-700 bg-gray-50 p-3 rounded-lg mb-4">
                {subject}
              </p>

              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-medium text-gray-900">Email Body:</p>
                <button
                  onClick={() => copyToClipboard(emailBody)}
                  className="text-xs text-gray-500 hover:text-black transition"
                >
                  Copy
                </button>
              </div>

              <div className="text-sm text-gray-700 max-h-96 overflow-y-auto px-4 py-4 whitespace-pre-wrap bg-gray-50 rounded-lg">
                {emailBody}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-gray-100 flex justify-between">
          <button
            onClick= {onClose}
            className="text-gray-500 hover:text-gray-700 transition"
          >
            Close
          </button>
          <button className="bg-black text-white px-4 py-2 rounded-full text-sm font-medium transition hover:bg-gray-800">
            Export All Emails
          </button>
        </div>
      </div>
    </div>
  );
};

export default EmailGeneratorModals;
