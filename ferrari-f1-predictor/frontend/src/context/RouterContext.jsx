import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

const RouterContext = createContext(null);

export function RouterProvider({ children }) {
  // Normalize initial path
  const getInitialPath = () => {
    const p = window.location.pathname.toLowerCase().replace(/\/$/, '') || '/';
    // Support hash fallback like /#championship
    if (window.location.hash) {
      const hashPath = window.location.hash.replace(/^#\/?/, '/');
      if (hashPath) return hashPath;
    }
    return p;
  };

  const [currentPath, setCurrentPath] = useState(getInitialPath);
  const [season, setSeason] = useState('2027');

  useEffect(() => {
    const handlePopState = () => {
      const p = window.location.pathname.toLowerCase().replace(/\/$/, '') || '/';
      setCurrentPath(p);
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  const navigate = useCallback((to) => {
    let cleanPath = to.toLowerCase().replace(/\/$/, '') || '/';
    if (!cleanPath.startsWith('/')) {
      cleanPath = '/' + cleanPath;
    }
    window.history.pushState({}, '', cleanPath);
    setCurrentPath(cleanPath);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, []);

  return (
    <RouterContext.Provider value={{ currentPath, navigate, season, setSeason }}>
      {children}
    </RouterContext.Provider>
  );
}

export function useRouter() {
  const context = useContext(RouterContext);
  if (!context) {
    throw new Error('useRouter must be used within a RouterProvider');
  }
  return context;
}

export function Link({ href, children, className = '', onClick, ...props }) {
  const { navigate, currentPath } = useRouter();
  const isActive = currentPath === (href.toLowerCase().replace(/\/$/, '') || '/');

  const handleClick = (e) => {
    if (onClick) onClick(e);
    if (!e.defaultPrevented && !e.metaKey && !e.ctrlKey && !e.shiftKey && !e.altKey && (!props.target || props.target === '_self')) {
      e.preventDefault();
      navigate(href);
    }
  };

  return (
    <a href={href} onClick={handleClick} className={typeof className === 'function' ? className({ isActive }) : className} {...props}>
      {children}
    </a>
  );
}
