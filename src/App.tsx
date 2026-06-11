import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { Layout } from './components/Layout';
import { LangProvider } from './lib/i18n';
import { Dashboard } from './pages/Dashboard';
import { GlossaryPage } from './pages/GlossaryPage';
import { LessonPage } from './pages/LessonPage';
import { PhasePage } from './pages/PhasePage';
import { ProgressPage } from './pages/ProgressPage';

export default function App() {
  return (
    <LangProvider>
      <BrowserRouter basename={import.meta.env.BASE_URL}>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/phase/:phaseSlug" element={<PhasePage />} />
            <Route path="/lesson/:phaseSlug/:lessonSlug" element={<LessonPage />} />
            <Route path="/glossary" element={<GlossaryPage />} />
            <Route path="/progress" element={<ProgressPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </LangProvider>
  );
}
