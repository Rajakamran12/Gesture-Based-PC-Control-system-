'use client';

import { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check localStorage for existing user session
    const storedUser = localStorage.getItem('gestureDashboardUser');
    if (storedUser) {
      try {
        setUser(JSON.parse(storedUser));
      } catch (e) {
        localStorage.removeItem('gestureDashboardUser');
      }
    }
    setLoading(false);
  }, []);

  const register = (email, password, name) => {
    // Get existing users from localStorage
    const usersJson = localStorage.getItem('gestureDashboardUsers');
    const users = usersJson ? JSON.parse(usersJson) : [];

    // Check if email already exists
    if (users.some(u => u.email === email)) {
      throw new Error('Email already registered');
    }

    // Create new user
    const newUser = {
      id: Date.now().toString(),
      email,
      password, // In production, hash this!
      name,
      createdAt: new Date().toISOString()
    };

    users.push(newUser);
    localStorage.setItem('gestureDashboardUsers', JSON.stringify(users));

    // Auto login after signup
    const sessionUser = { id: newUser.id, email, name };
    localStorage.setItem('gestureDashboardUser', JSON.stringify(sessionUser));
    setUser(sessionUser);
  };

  const login = (email, password) => {
    const usersJson = localStorage.getItem('gestureDashboardUsers');
    const users = usersJson ? JSON.parse(usersJson) : [];

    const foundUser = users.find(u => u.email === email && u.password === password);
    if (!foundUser) {
      throw new Error('Invalid email or password');
    }

    const sessionUser = { id: foundUser.id, email: foundUser.email, name: foundUser.name };
    localStorage.setItem('gestureDashboardUser', JSON.stringify(sessionUser));
    setUser(sessionUser);
  };

  const logout = () => {
    localStorage.removeItem('gestureDashboardUser');
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, loading, register, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}
