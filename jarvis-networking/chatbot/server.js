// Prompt Refinement System
// This system analyzes a user's initial search criteria and generates confirmation questions
// to refine the search parameters for better matching results.

// Configuration for Gemini API
require('dotenv').config(); 
const { GoogleGenerativeAI } = require('@google/generative-ai');
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Main function to refine user prompts
async function refineUserPrompt(userInput) {
    try {
      // Get the Gemini model
      const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
  
      // Create the system prompt with clear instructions
      const systemPrompt = `
      You are a professional networking assistant that helps refine search criteria. 
      Analyze the user's search input and generate ONE specific follow-up question to clarify their prompt for us to process keyword matching more efficiently.
      
      Focus on these aspects:
      1. If they mention industries or functions, ask them to list out specific company names.
      2. If they mention a company, ask about specific roles or departments.
      3. If they mention educational background, ask if they want to filter by specific institutions.
      4. If they mention experience level, ask about specific years of experience.
      5. If they mention location, ask if they want remote or in-person connections.
      6. If they mention skills, ask if they prioritize technical or soft skills.
      
      IMPORTANT: Return ONLY a JSON object with these fields without any markdown formatting or code blocks:
      {
        "confirmationQuestion": "Your follow-up question here",
        "options": ["Option 1", "Option 2", "..."] (2-4 possible answers if applicable, otherwise an empty array),
        "refinementType": "One of: industry, company, role, education, experience, location, skills"
      }
      
      Keep questions concise and focused on improving search results.
      `;
  
      // Combine system prompt with user input
      const promptParts = [
        { text: systemPrompt },
        { text: `User search criteria: "${userInput}"` }
      ];
  
      // Generate the completion
      const result = await model.generateContent({
        contents: [{ role: "user", parts: promptParts }],
        generationConfig: {
          temperature: 0.2,
        },
      });
  
      // Parse the response
      const responseText = result.response.text();
      
      // Extract JSON from the response in case it's wrapped in markdown code blocks
      let jsonText = responseText;
      
      // Check if the response is wrapped in a code block
      const jsonMatch = responseText.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
      if (jsonMatch) {
        jsonText = jsonMatch[1];
      }
      
      try {
        const jsonResponse = JSON.parse(jsonText);
        return jsonResponse;
      } catch (parseError) {
        console.error("Error parsing JSON response:", parseError);
        console.log("Response received:", responseText);
        
        // Fallback response
        return {
          confirmationQuestion: "Which specific companies are you interested in applying to for software engineering roles?",
          options: ["Big Tech (FAANG)", "Startups", "Local companies", "Any company"],
          refinementType: "company"
        };
      }
    } catch (error) {
      console.error("Error refining prompt:", error);
      console.log("For input:", userInput);
    }
  }

// Function to apply user's confirmation response to the original prompt
async function applyRefinement(originalPrompt, refinementType, userResponse) {
    try {
      // Get the Gemini model
      const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
  
      // Create the system prompt with clear instructions
      const systemPrompt = `
      You are a professional networking assistant that helps refine search criteria.
      Based on the original search prompt and the user's response to your clarification question,
      create a refined, structured search prompt that will yield better matching results.
      
      IMPORTANT: Return ONLY a JSON object with these fields without any markdown formatting or code blocks:
      {
        "refinedPrompt": "The improved search prompt",
        "searchParams": {
          "keywords": ["keyword1", "keyword2"],
          "industry": ["industry1", "industry2"],
          "company": ["company1", "company2"],
          "role": ["role1", "role2"],
          "education": ["education1", "education2"],
          "experience": ["experience1", "experience2"],
          "location": ["location1", "location2"],
          "skills": ["skill1", "skill2"]
        }
      }
      
      Fill in relevant parameters based on available information, leave others as empty arrays.
      Make the refinedPrompt clear, specific, and comprehensive.
      `;
  
      // Combine system prompt with context
      const promptParts = [
        { text: systemPrompt },
        { text: `Original search criteria: "${originalPrompt}"` },
        { text: `Refinement type: ${refinementType}` },
        { text: `User's response to clarification: "${userResponse}"` }
      ];
  
      // Generate the completion
      const result = await model.generateContent({
        contents: [{ role: "user", parts: promptParts }],
        generationConfig: {
          temperature: 0.2,
        },
      });
  
      // Parse the response
      const responseText = result.response.text();
      
      // Extract JSON from the response in case it's wrapped in markdown code blocks
      let jsonText = responseText;
      
      // Check if the response is wrapped in a code block
      const jsonMatch = responseText.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
      if (jsonMatch) {
        jsonText = jsonMatch[1];
      }
      
      try {
        const jsonResponse = JSON.parse(jsonText);
        return jsonResponse;
      } catch (parseError) {
        console.error("Error parsing JSON response:", parseError);
        console.log("Response received:", responseText);
        
        // Fallback response
        return {
          refinedPrompt: `${originalPrompt} looking for ${userResponse}`,
          searchParams: {
            keywords: ["software engineering", "SWE"],
            industry: ["technology"],
            company: [],
            role: ["software engineer"],
            education: ["University of Waterloo"],
            experience: [],
            location: [],
            skills: []
          }
        };
      }
    } catch (error) {
      console.error("Error applying refinement:", error);
      return {
        refinedPrompt: `${originalPrompt} with preference for ${userResponse}`,
        searchParams: {
          keywords: ["software engineering", "SWE"],
          industry: ["technology"],
          company: [],
          role: ["software engineer"],
          education: ["University of Waterloo"],
          experience: [],
          location: [],
          skills: []
        }
      };
    }
}

// Example implementation for a simple API endpoint (using Express)
// You'll need to: npm install express cors body-parser
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
app.use(cors());
app.use(bodyParser.json());

// Endpoint to get confirmation questions
app.post('/api/refine-prompt', async (req, res) => {
  const { userInput } = req.body;
  
  if (!userInput) {
    return res.status(400).json({ error: 'Missing user input' });
  }
  
  try {
    const refinementQuestion = await refineUserPrompt(userInput);
    res.json(refinementQuestion);
  } catch (error) {
    console.error('Error processing request:', error);
    res.status(500).json({ error: 'Failed to process request' });
  }
});

// Endpoint to apply refinement
app.post('/api/apply-refinement', async (req, res) => {
  const { originalPrompt, refinementType, userResponse } = req.body;
  
  if (!originalPrompt || !refinementType || !userResponse) {
    return res.status(400).json({ error: 'Missing required parameters' });
  }
  
  try {
    const refinedPrompt = await applyRefinement(originalPrompt, refinementType, userResponse);
    res.json(refinedPrompt);
  } catch (error) {
    console.error('Error processing refinement:', error);
    res.status(500).json({ error: 'Failed to process refinement' });
  }
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Prompt refinement service running on port ${PORT}`);
});

// Export functions for testing/importing
module.exports = {
  refineUserPrompt,
  applyRefinement
};