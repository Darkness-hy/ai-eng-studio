import { lazy, Suspense } from 'react';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { Layout } from './components/Layout';
import { AuthProvider } from './lib/auth';
import { LangProvider } from './lib/i18n';
import { Dashboard } from './pages/Dashboard';
import { GlossaryPage } from './pages/GlossaryPage';
import { LessonPage } from './pages/LessonPage';
import { LoginPage } from './pages/LoginPage';
import { PhasePage } from './pages/PhasePage';
import { ProgressPage } from './pages/ProgressPage';

const AdminPage = lazy(() => import('./pages/AdminPage').then((m) => ({ default: m.AdminPage })));

export default function App() {
  return (
    <LangProvider>
      <AuthProvider>
        <BrowserRouter basename={import.meta.env.BASE_URL}>
          <Routes>
            <Route element={<Layout />}>
              <Route path="/" element={<Dashboard />} />
              <Route path="/phase/:phaseSlug" element={<PhasePage />} />
              <Route path="/lesson/:phaseSlug/:lessonSlug" element={<LessonPage />} />
              <Route path="/glossary" element={<GlossaryPage />} />
              <Route path="/progress" element={<ProgressPage />} />
              <Route path="/login" element={<LoginPage />} />
              <Route
                path="/admin"
                element={
                  <Suspense
                    fallback={<div className="py-32 text-center text-faint">…</div>}
                  >
                    <AdminPage />
                  </Suspense>
                }
              />
            </Route>
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </LangProvider>
  );
}
