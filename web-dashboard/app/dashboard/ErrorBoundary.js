'use client';

import React from 'react';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error('ErrorBoundary caught:', error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '100vh',
          background: '#0f0f0f',
          color: '#f0f0f0',
          fontFamily: '"Segoe UI", sans-serif',
          padding: 20
        }}>
          <div style={{maxWidth:720,textAlign:'center'}}>
            <h2 style={{marginBottom:8}}>Something went wrong</h2>
            <p style={{color:'#cfcfcf',marginBottom:16}}>An error occurred while loading the dashboard. The issue has been logged to the console.</p>
            <div style={{display:'flex',gap:8,justifyContent:'center'}}>
              <button onClick={() => location.reload()} style={{padding:'8px 12px',background:'#16a34a',border:'none',color:'#fff',borderRadius:6}}>Reload</button>
              <button onClick={() => { localStorage.removeItem('gestureDashboardUser'); location.href = '/login'; }} style={{padding:'8px 12px',background:'#374151',border:'none',color:'#fff',borderRadius:6}}>Sign out</button>
            </div>
            <pre style={{textAlign:'left',marginTop:16,whiteSpace:'pre-wrap',color:'#ffdddd'}}>{String(this.state.error && this.state.error.message)}</pre>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
