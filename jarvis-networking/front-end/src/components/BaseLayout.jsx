import React, { useEffect, useState } from 'react';
import axios from 'axios';

const BaseLayout = ({ children }) => {
  const [user, setUser] = useState(null);

  useEffect(() => {
    // 🔧 Hardwired mock user for dev purposes
    setUser({
      name: "Dev User",
      email: "dev@example.com",
      picture: "https://via.placeholder.com/150"
    });
   // Optional real call (commented out for now)
/*
   axios.get('/auth/user')
       .then(res => setUser(res.data))
       .catch(() => setUser(null));
*/
       }, []);
  
  const login = () => window.location.href = 'http://localhost:8000/auth/login';
  const logout = () => window.location.href = 'http://localhost:8000/auth/logout';

  return (
    <div className="flex flex-col w-full max-w-5xl mx-auto bg-white min-h-screen">
      <header className="sticky top-0 z-10 bg-white bg-opacity-90 backdrop-blur-md border-b border-gray-100 px-8 py-4">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-medium text-gray-900">JARVIS Networking</h1>
          <div className="flex items-center space-x-6">
            {!user ? (
              <button onClick={login} className="text-sm font-medium text-gray-500 hover:text-black">
                Sign in with Google
              </button>
            ) : (
              <>
                <span className="text-sm text-gray-700">{user.name}</span>
                <img src={user.picture} alt="avatar" className="h-8 w-8 rounded-full" />
                <button onClick={logout} className="text-sm text-gray-500 hover:text-black">
                  Logout
                </button>
              </>
            )}
          </div>
        </div>
      </header>

      <main className="flex-1 px-8 py-12 w-full">
        {user ? children : <p className="text-center text-gray-500">Please log in to use this tool.</p>}
      </main>

      <footer className="border-t border-gray-100 px-8 py-6">
        <div className="flex justify-between items-center text-sm text-gray-500">
          <div>© 2025 JARVIS Foundation </div>
          <div className="flex space-x-6">
            <span className="cursor-pointer hover:text-black">Privacy</span>
            <span className="cursor-pointer hover:text-black">Terms</span>
            <span className="cursor-pointer hover:text-black">Contact</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default BaseLayout;
