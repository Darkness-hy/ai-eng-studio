import { lazy, Suspense, type ReactNode } from 'react';
import { BrowserRouter, Navigate, Route, Routes, useLocation } from 'react-router-dom';
import { Layout } from './components/Layout';
import { AuthProvider, useAuth } from './lib/auth';
import { LangProvider } from './lib/i18n';
import { Dashboard } from './pages/Dashboard';
import { GlossaryPage } from './pages/GlossaryPage';
import { LessonPage } from './pages/LessonPage';
import { LeaderboardsPage } from './pages/LeaderboardsPage';
import { LoginPage } from './pages/LoginPage';
import { PhasePage } from './pages/PhasePage';
import { PlacementPage } from './pages/PlacementPage';
import { ProgressPage } from './pages/ProgressPage';
import { ClassDetailPage } from './pages/ClassDetailPage';
import { ClassesPage } from './pages/ClassesPage';
import { FlashcardsPage } from './pages/FlashcardsPage';
import { ReviewPage } from './pages/ReviewPage';

const AdminPage = lazy(() => import('./pages/AdminPage').then((m) => ({ default: m.AdminPage })));

/** With cloud sync configured, the site requires sign-in: unauthenticated
 *  visitors land on /login first. Without configuration it stays local-only. */
function Gate({ children }: { children: ReactNode }) {
  const { enabled, loading, profile } = useAuth();
  const location = useLocation();
  if (!enabled) return children;
  if (loading) {
    return <div className="flex min-h-dvh items-center justify-center text-faint">…</div>;
  }
  if (!profile && location.pathname !== '/login') {
    return <Navigate to="/login" replace />;
  }
  return children;
}

export default function App() {
  return (
    <LangProvider>
      <AuthProvider>
        <BrowserRouter basename={import.meta.env.BASE_URL}>
          <Gate>
            <Routes>
              <Route element={<Layout />}>
                <Route path="/" element={<Dashboard />} />
                <Route path="/phase/:phaseSlug" element={<PhasePage />} />
                <Route path="/lesson/:phaseSlug/:lessonSlug" element={<LessonPage />} />
                <Route path="/glossary" element={<GlossaryPage />} />
                <Route path="/progress" element={<ProgressPage />} />
                <Route path="/leaderboards" element={<LeaderboardsPage />} />
                <Route path="/find-your-level" element={<PlacementPage />} />
                <Route path="/review" element={<ReviewPage />} />
                <Route path="/flashcards" element={<FlashcardsPage />} />
                <Route path="/classes" element={<ClassesPage />} />
                <Route path="/class/:classId" element={<ClassDetailPage />} />
                <Route path="/login" element={<LoginPage />} />
                <Route
                  path="/admin"
                  element={
                    <Suspense fallback={<div className="py-32 text-center text-faint">…</div>}>
                      <AdminPage />
                    </Suspense>
                  }
                />
              </Route>
            </Routes>
          </Gate>
        </BrowserRouter>
      </AuthProvider>
    </LangProvider>
  );
}
