import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, 
  Brain, 
  ShieldAlert, 
  Heart, 
  Languages, 
  RefreshCw, 
  BarChart2, 
  User, 
  Users, 
  Upload, 
  Send, 
  MessageSquare,
  AlertTriangle,
  Info,
  Clock,
  ChevronRight,
  ChevronLeft
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

// Simple Plotly Wrapper to render CDN charts
const PlotlyChart = ({ id, data, layout }) => {
  useEffect(() => {
    if (window.Plotly && data && layout) {
      window.Plotly.newPlot(id, data, {
        ...layout,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Plus Jakarta Sans, sans-serif' },
        margin: { t: 40, r: 20, l: 40, b: 40 },
        responsive: true
      });
    }
  }, [id, data, layout]);

  return <div id={id} className="chart-surface" style={{ width: '100%', height: '360px' }} />;
};

// Lightweight Markdown & Plain-Text Formatter to prevent raw tags displaying in UI
const renderTextFormat = (text) => {
  if (!text) return '';
  
  // Escape HTML tags to prevent XSS
  let escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  
  // Bolding (**word** or *word*)
  escaped = escaped.replace(/\*\*(.*?)\*\*/g, '<strong style="color: var(--primary-brand); font-weight: 700;">$1</strong>');
  escaped = escaped.replace(/\*(.*?)\*/g, '<strong style="color: var(--primary-brand); font-weight: 700;">$1</strong>');
  
  // Clean markdown headings: ## Heading or ### Heading
  escaped = escaped.replace(/^##\s+(.*$)/gim, '<div style="font-weight: 800; color: var(--primary-brand); font-size: 1.1rem; margin: 16px 0 8px; text-transform: uppercase;">$1</div>');
  escaped = escaped.replace(/^###\s+(.*$)/gim, '<div style="font-weight: 700; color: var(--primary-brand); font-size: 1rem; margin: 12px 0 6px;">$1</div>');
  escaped = escaped.replace(/^#\s+(.*$)/gim, '<div style="font-weight: 800; color: var(--primary-brand); font-size: 1.3rem; margin: 20px 0 10px;">$1</div>');
  
  // Lists
  escaped = escaped.replace(/^\* (.*$)/gim, '<li style="margin-left: 16px; margin-bottom: 6px;">$1</li>');
  escaped = escaped.replace(/^- (.*$)/gim, '<li style="margin-left: 16px; margin-bottom: 6px;">$1</li>');
  
  // Horizontal rules
  escaped = escaped.replace(/^---\s*$/gim, '<hr style="border: none; border-top: 1px solid var(--border-light); margin: 16px 0;" />');
  
  // Newlines to line breaks
  escaped = escaped.replace(/\n/g, '<br />');

  return <div dangerouslySetInnerHTML={{ __html: escaped }} />;
};

function App() {
  const [activeMode, setActiveMode] = useState('patient'); // 'patient' or 'community'
  const [language, setLanguage] = useState('English');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  
  // Patient Mode States
  const [patientSession, setPatientSession] = useState(null);
  const [activeReportIdx, setActiveReportIdx] = useState(0);
  const [chatHistory, setChatHistory] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatSending, setIsChatSending] = useState(false);
  const [generatingExplanation, setGeneratingExplanation] = useState(false);
  const [patientExplanations, setPatientExplanations] = useState({}); // keyed by report_id

  // Community Mode States
  const [communityData, setCommunityData] = useState(null);
  const [loadingCommunity, setLoadingCommunity] = useState(false);
  const [selectedRegion, setSelectedRegion] = useState('Auto-assign (random)');
  const [selectedAgeGroup, setSelectedAgeGroup] = useState('Auto-assign (random)');
  const [selectedTrendTest, setSelectedTrendTest] = useState('');
  const [trendDays, setTrendDays] = useState(30);
  const [trendData, setTrendData] = useState(null);
  const [loadingTrend, setLoadingTrend] = useState(false);
  const [communityQuestion, setCommunityQuestion] = useState('');
  const [communityAnswer, setCommunityAnswer] = useState('');
  const [loadingCommChat, setLoadingCommChat] = useState(false);
  const [availableTests, setAvailableTests] = useState([]);
  const [useDP, setUseDP] = useState(true);

  const chatEndRef = useRef(null);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  // Fetch Community Data and available tests
  const fetchCommunityDashboard = async () => {
    setLoadingCommunity(true);
    try {
      const res = await fetch(`${API_BASE}/community/dashboard?use_dp=${useDP}`);
      const data = await res.json();
      setCommunityData(data);
      
      // Load available test names
      const tRes = await fetch(`${API_BASE}/community/tests`);
      const tData = await tRes.json();
      setAvailableTests(tData.tests || []);
      if (tData.tests && tData.tests.length > 0 && !selectedTrendTest) {
        setSelectedTrendTest(tData.tests[0]);
      }
    } catch (e) {
      console.error("Failed to load community dashboard", e);
    } finally {
      setLoadingCommunity(false);
    }
  };

  useEffect(() => {
    if (activeMode === 'community') {
      fetchCommunityDashboard();
    }
  }, [activeMode, useDP]);

  // Load Trend when test or days change
  useEffect(() => {
    const fetchTrend = async () => {
      if (!selectedTrendTest || activeMode !== 'community') return;
      setLoadingTrend(true);
      try {
        const res = await fetch(`${API_BASE}/community/trends?test_name=${encodeURIComponent(selectedTrendTest)}&days_ahead=${trendDays}`);
        const data = await res.json();
        setTrendData(data);
      } catch (e) {
        console.error("Failed to load trend", e);
      } finally {
        setLoadingTrend(false);
      }
    };
    fetchTrend();
  }, [selectedTrendTest, trendDays, activeMode]);

  // Patient Mode Single/Multi File Upload
  const handlePatientUpload = async (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    
    setIsUploading(true);
    setUploadStatus('Uploading and parsing reports...');
    
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }
    formData.append('mode', 'patient');
    
    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setPatientSession(data);
      setActiveReportIdx(0);
      setChatHistory([
        { sender: 'assistant', text: `Hello! I have successfully ingested ${files.length} report(s). I have extracted all diagnostic parameters and cataloged them into FHIR Observations. Ask me any questions about your results!` }
      ]);
    } catch (err) {
      alert("Failed to upload file. Check if backend is active.");
      console.error(err);
    } finally {
      setIsUploading(false);
      setUploadStatus('');
    }
  };

  // Community Mode Batch File Upload
  const handleCommunityUpload = async (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    
    setIsUploading(true);
    setUploadStatus(`Uploading ${files.length} reports to database...`);
    
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }
    formData.append('region', selectedRegion);
    formData.append('age_group', selectedAgeGroup);
    formData.append('mode', 'community');
    
    try {
      await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });
      alert("Batch reports successfully ingested and anonymized!");
      fetchCommunityDashboard();
    } catch (err) {
      alert("Failed to upload batch.");
      console.error(err);
    } finally {
      setIsUploading(false);
      setUploadStatus('');
    }
  };

  // Generate Patient-Friendly Summary (with language support)
  const generatePatientFriendlyExplanation = async (report) => {
    setGeneratingExplanation(true);
    try {
      // We will ask the QA Agent directly using a specific prompt for localized structured summary
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: `Generate a patient-friendly summary of this medical report in ${language}. Ensure the output contains: 
1. Overview: brief explanation of what the report is
2. What Needs Attention: explanation of abnormal findings (in plain terms, why they matter)
3. What Looks Good: explanation of normal parameters
4. Recommended Next Steps (dietary, clinical recommendation to consult doctor)
Do not use emojis in descriptions.`,
          collection_name: patientSession.session_id,
          combined_text: report.raw_text,
          language: language
        })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let textBuffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6).trim();
            if (dataStr === '[DONE]') continue;
            try {
              const data = JSON.parse(dataStr);
              if (data.type === 'chunk') {
                textBuffer += data.text;
                // Update local summary state continuously for streaming feel
                setPatientExplanations(prev => ({
                  ...prev,
                  [report.report_id]: textBuffer
                }));
              }
            } catch (e) {}
          }
        }
      }
    } catch (e) {
      console.error(e);
    } finally {
      setGeneratingExplanation(false);
    }
  };

  // SSE Patient Chat streaming
  const handleSendChat = async () => {
    if (!chatInput.trim() || !patientSession) return;
    
    const query = chatInput;
    setChatInput('');
    setChatHistory(prev => [...prev, { sender: 'user', text: query }]);
    setIsChatSending(true);

    const activeReport = patientSession.results[activeReportIdx];
    
    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          collection_name: patientSession.session_id,
          language: language
        })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      // Add initial empty assistant bubble
      setChatHistory(prev => [...prev, { sender: 'assistant', text: '', sources: [] }]);

      let answerText = '';
      
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6).trim();
            if (dataStr === '[DONE]') continue;
            try {
              const data = JSON.parse(dataStr);
              if (data.type === 'chunk') {
                answerText += data.text;
                setChatHistory(prev => {
                  const updated = [...prev];
                  updated[updated.length - 1].text = answerText;
                  return updated;
                });
              } else if (data.type === 'sources') {
                setChatHistory(prev => {
                  const updated = [...prev];
                  updated[updated.length - 1].sources = data.sources || [];
                  return updated;
                });
              }
            } catch (e) {}
          }
        }
      }
    } catch (err) {
      console.error(err);
      setChatHistory(prev => [...prev, { sender: 'assistant', text: 'Error connecting to RAG chatbot server.' }]);
    } finally {
      setIsChatSending(false);
    }
  };

  // Community Mode Ask AI Query
  const handleCommunityChat = async () => {
    if (!communityQuestion.trim()) return;
    setLoadingCommChat(true);
    try {
      const res = await fetch(`${API_BASE}/community/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: communityQuestion })
      });
      const data = await res.json();
      setCommunityAnswer(data.answer);
    } catch (e) {
      setCommunityAnswer("Error fetching database analysis. Please try again.");
    } finally {
      setLoadingCommChat(false);
    }
  };

  // Prepare plots for the charts
  const getFlagDistributionChart = () => {
    if (!communityData || !communityData.flag_distribution) return null;
    const labels = Object.keys(communityData.flag_distribution);
    const values = Object.values(communityData.flag_distribution);
    const colors = labels.map(l => {
      if (l === 'NORMAL') return '#16c79e';
      if (l === 'HIGH' || l === 'LOW') return '#f39c12';
      return '#e83a30';
    });

    return {
      data: [{
        values: values,
        labels: labels,
        type: 'pie',
        hole: 0.45,
        marker: { colors: colors },
        textinfo: 'label+percent',
        hoverinfo: 'label+value'
      }],
      layout: {
        title: 'Flag Severity Distribution',
        height: 360,
        showlegend: false
      }
    };
  };

  const getTopAbnormalChart = () => {
    if (!communityData || !communityData.top_abnormal) return null;
    const x = communityData.top_abnormal.map(t => t.abnormal_count);
    const y = communityData.top_abnormal.map(t => t.test_name);

    return {
      data: [{
        x: x,
        y: y,
        type: 'bar',
        orientation: 'h',
        marker: { color: '#104891' }
      }],
      layout: {
        title: 'Top Abnormal Parameters',
        height: 360,
        xaxis: { title: 'Number of Occurrences' },
        yaxis: { automargin: true }
      }
    };
  };

  const getTrendChart = () => {
    if (!trendData) return null;
    
    // Historical Series
    const histDates = trendData.historical.map(h => h.date);
    const histRates = trendData.historical.map(h => h.abnormal_rate);
    
    // Forecast Series
    const foreDates = trendData.forecast.map(f => f.date);
    const foreRates = trendData.forecast.map(f => f.abnormal_rate);

    return {
      data: [
        {
          x: histDates,
          y: histRates,
          name: 'Historical Anomaly %',
          type: 'scatter',
          mode: 'lines+markers',
          line: { color: '#104891', width: 3 }
        },
        {
          x: foreDates,
          y: foreRates,
          name: `Forecast (${trendDays}d)`,
          type: 'scatter',
          mode: 'lines',
          line: { color: '#e83a30', dash: 'dot', width: 3 }
        }
      ],
      layout: {
        title: `${selectedTrendTest.toUpperCase()} Anomaly Trend & Forecast`,
        height: 360,
        xaxis: { title: 'Timeline' },
        yaxis: { title: 'Abnormal Rate (%)', range: [0, 100] }
      }
    };
  };

  const getHeatmapChart = () => {
    if (!communityData || !communityData.demographic_cross_tab) return null;
    
    const cross = communityData.demographic_cross_tab;
    const testNames = Object.keys(cross);
    if (testNames.length === 0) return null;
    
    // We'll plot the first available test in cross-tab or map it dynamically
    const targetTest = selectedTrendTest.toLowerCase();
    const testCross = cross[targetTest] || cross[Object.keys(cross)[0]];
    if (!testCross) return null;
    
    const regions = testCross.regions || [];
    const ageGroups = testCross.age_groups || [];
    const z = testCross.matrix || [];

    return {
      data: [{
        z: z,
        x: regions,
        y: ageGroups,
        type: 'heatmap',
        colorscale: [
          [0.0, '#ffffff'],
          [0.3, '#16c79e-soft'],
          [0.6, '#f39c12'],
          [1.0, '#e83a30']
        ].map(([v, c]) => [v, c === '#16c79e-soft' ? 'rgba(22, 199, 158, 0.4)' : c]),
        hoverongaps: false
      }],
      layout: {
        title: `Demographic Risk Index: ${targetTest.toUpperCase()}`,
        height: 360,
        xaxis: { title: 'Locality / Region' },
        yaxis: { title: 'Age Group' }
      }
    };
  };

  const activeReport = patientSession?.results?.[activeReportIdx];

  return (
    <div className="stApp-container">
      {/* Floating Toggle Button */}
      <button 
        className={`sidebar-toggle-btn ${sidebarCollapsed ? 'collapsed' : ''}`}
        onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
        title={sidebarCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
      >
        {sidebarCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
      </button>

      {/* Sidebar Navigation */}
      <aside className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
        <div className="sidebar-title">COMMUNITY HEALTH</div>
        <div className="sidebar-subtitle">Intelligence Workspace</div>

        <nav className="nav-menu">
          <div 
            className={`nav-item ${activeMode === 'patient' ? 'active' : ''}`}
            onClick={() => setActiveMode('patient')}
          >
            <User size={18} />
            <span>Patient Workspace</span>
          </div>
          
          <div 
            className={`nav-item ${activeMode === 'community' ? 'active' : ''}`}
            onClick={() => setActiveMode('community')}
          >
            <Users size={18} />
            <span>Community Dashboard</span>
          </div>
        </nav>

        <div className="sidebar-divider" />
        
        {/* Global Settings */}
        <div className="form-group" style={{ marginBottom: '20px' }}>
          <label className="form-label">
            <Languages size={14} style={{ marginRight: '6px', verticalAlign: 'text-bottom' }} />
            App Language
          </label>
          <select 
            className="form-select"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
          >
            <option value="English">English</option>
            <option value="Hindi">Hindi (हिन्दी)</option>
            <option value="Bengali">Bengali (বাংলা)</option>
            <option value="Telugu">Telugu (తెలుగు)</option>
            <option value="Tamil">Tamil (தமிழ்)</option>
          </select>
        </div>

        {activeMode === 'community' && (
          <div className="form-group">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <label className="form-label">Differential Privacy</label>
              <input 
                type="checkbox" 
                checked={useDP} 
                onChange={(e) => setUseDP(e.target.checked)}
                style={{ cursor: 'pointer' }}
              />
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              Protects patients with Laplace noise ($\epsilon = 0.5$)
            </div>
          </div>
        )}
      </aside>

      {/* Main Panel */}
      <main className="dashboard-main">
        {/* Hero Section */}
        <div className="app-hero">
          <h1>Community Health Intelligence Assistant</h1>
          <p>
            Bridging health literacy barriers with source-grounded RAG intelligence and protecting clinics with Differential Privacy algorithms.
          </p>
        </div>

        {/* Medical Disclaimer Banner */}
        <div className="disclaimer-bar">
          <strong>⚠️ Clinician Advisory & Patient Warning:</strong> This system uses generative AI models for information summarization and clinical trend projections. It does not provide medical diagnoses or replace direct consultation with professional healthcare practitioners. Always review raw parameters.
        </div>

        {isUploading && (
          <div className="alert-card warning">
            <RefreshCw className="animate-spin" size={18} />
            <div>
              <strong>Processing:</strong> {uploadStatus}
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* PATIENT WORKSPACE VIEW                                       */}
        {/* ============================================================ */}
        {activeMode === 'patient' && (
          <div>
            {/* Upload Area */}
            {!patientSession && (
              <div className="upload-dropzone">
                <Upload size={40} style={{ color: '#104891', marginBottom: '12px' }} />
                <h3>Upload Your Diagnostic Lab Reports</h3>
                <p style={{ color: '#5e6b7c', fontSize: '0.9rem' }}>
                  Supports single or multiple PDF documents. Parsed into FHIR Observables.
                </p>
                <input 
                  type="file" 
                  accept=".pdf" 
                  multiple 
                  onChange={handlePatientUpload}
                  style={{ display: 'none' }}
                  id="patient-file-input"
                />
                <button 
                  className="primary-button" 
                  onClick={() => document.getElementById('patient-file-input').click()}
                  style={{ marginTop: '16px' }}
                >
                  Browse PDF Files
                </button>
              </div>
            )}

            {/* Ingested Sessions Dashboard */}
            {patientSession && (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                  <h3 style={{ margin: 0 }}>Patient Diagnostics Overview</h3>
                  <button 
                    className="primary-button" 
                    onClick={() => { setPatientSession(null); setChatHistory([]); }}
                    style={{ background: '#e5d5c5', color: '#242424', boxShadow: 'none' }}
                  >
                    Clear and Upload New
                  </button>
                </div>

                {/* Tabs for Multiple Reports */}
                {patientSession.results.length > 1 && (
                  <div className="tabs-container">
                    {patientSession.results.map((res, idx) => (
                      <button
                        key={res.report_id}
                        className={`tab-button ${activeReportIdx === idx ? 'active' : ''}`}
                        onClick={() => { setActiveReportIdx(idx); }}
                      >
                        {res.filename}
                      </button>
                    ))}
                  </div>
                )}

                {activeReport && (
                  <div>
                    {/* Metrics Row */}
                    <div className="stats-grid">
                      <div className="metric-card">
                        <div className="value">{activeReport.fhir_observations.length}</div>
                        <div className="label">Total Test Observations</div>
                      </div>
                      <div className="metric-card">
                        <div className="value" style={{ color: activeReport.risk_summary.normal > 0 ? '#16c79e' : '#242424' }}>
                          {activeReport.risk_summary.normal}
                        </div>
                        <div className="label">Normal Results</div>
                      </div>
                      <div className="metric-card">
                        <div className="value" style={{ color: activeReport.risk_summary.abnormal > 0 ? '#f39c12' : '#242424' }}>
                          {activeReport.risk_summary.abnormal}
                        </div>
                        <div className="label">Abnormal Flags</div>
                      </div>
                      <div className="metric-card">
                        <div className="value" style={{ color: activeReport.risk_summary.critical > 0 ? '#e83a30' : '#242424' }}>
                          {activeReport.risk_summary.critical}
                        </div>
                        <div className="label">Critical Bounds</div>
                      </div>
                    </div>

                    {/* Risk Severity Card */}
                    {(() => {
                      const score = activeReport.risk_summary.risk_score;
                      let riskClass = 'risk-card-normal';
                      let riskLabel = 'Normal Health Profile';
                      let riskDesc = 'All analyzed parameters are within normal physiological bounds. Maintain your current wellness path.';
                      
                      if (score > 1.5) {
                        riskClass = 'risk-card-critical';
                        riskLabel = 'CRITICAL ALERTS DETECTED';
                        riskDesc = 'One or more parameters reside within critical bounds. Review immediate details below and consult your doctor.';
                      } else if (score > 0.5) {
                        riskClass = 'risk-card-elevated';
                        riskLabel = 'ELEVATED CLINICAL RISKS';
                        riskDesc = 'Minor health deviations observed. Review dietary patterns and schedule a follow-up assessment.';
                      } else if (score > 0) {
                        riskClass = 'risk-card-mild';
                        riskLabel = 'MILD DEVIATIONS';
                        riskDesc = 'Slight deviations detected, mostly within high or low buffer bounds.';
                      }

                      return (
                        <div className={`risk-card ${riskClass}`}>
                          <h3 style={{ margin: '0 0 6px 0', fontSize: '1.2rem' }}>{riskLabel}</h3>
                          <p style={{ margin: 0, fontSize: '0.92rem', opacity: 0.9 }}>{riskDesc}</p>
                        </div>
                      );
                    })()}

                    {/* Lab values display */}
                    <h3>Extracted Parameters (FHIR R4 Format)</h3>
                    <div className="lab-grid">
                      {activeReport.fhir_observations.map((obs) => {
                        const interpretation = obs.interpretation?.[0]?.text || 'NORMAL';
                        let cardClass = 'lab-card-normal';
                        let badgeClass = 'flag-normal';

                        if (interpretation.includes('CRITICAL')) {
                          cardClass = 'lab-card-critical_high';
                          badgeClass = 'flag-critical';
                        } else if (interpretation.includes('HIGH') || interpretation.includes('LOW')) {
                          cardClass = 'lab-card-high';
                          badgeClass = 'flag-high';
                        }

                        return (
                          <div key={obs.id} className={`lab-card ${cardClass}`}>
                            <div className="header-row">
                              <span className="name">{obs.code.text}</span>
                              <span className={`flag-badge ${badgeClass}`}>{interpretation}</span>
                            </div>
                            <div className="value">
                              {obs.valueQuantity.value} <span style={{ fontSize: '0.85rem', fontWeight: 500 }}>{obs.valueQuantity.unit}</span>
                            </div>
                            {obs.referenceRange?.[0] && (
                              <div className="ref-range">
                                Normal: {obs.referenceRange[0].text}
                              </div>
                            )}
                            <div style={{ fontSize: '0.7rem', color: '#5e6b7c', marginTop: '8px' }}>
                              LOINC: {obs.code.coding[0].code}
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    {/* AI Translation & Plain-language summaries */}
                    <div style={{ marginTop: '28px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                        <h3>AI Plain-Language Summary ({language})</h3>
                        <button
                          className="primary-button"
                          onClick={() => generatePatientFriendlyExplanation(activeReport)}
                          disabled={generatingExplanation}
                        >
                          {generatingExplanation ? 'Translating...' : `Generate Friendly Explanation`}
                        </button>
                      </div>

                      {patientExplanations[activeReport.report_id] && (
                        <div 
                          className="metric-card" 
                          style={{ lineHeight: 1.7, fontSize: '0.96rem', background: '#ffffff', borderLeft: '4px solid #16c79e' }}
                        >
                          {renderTextFormat(patientExplanations[activeReport.report_id])}
                        </div>
                      )}
                    </div>

                    {/* Interactive RAG chat */}
                    <div className="chat-container">
                      <h3 style={{ marginTop: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Brain size={20} />
                        Ask Questions About Your Report
                      </h3>
                      <div className="chat-history">
                        {chatHistory.map((msg, idx) => {
                          const isLast = idx === chatHistory.length - 1;
                          const showTyping = isLast && msg.sender === 'assistant' && !msg.text && isChatSending;

                          return (
                            <div key={idx} className={`chat-bubble ${msg.sender}`}>
                              {showTyping ? (
                                <div className="typing-indicator">
                                  <span className="typing-dot"></span>
                                  <span className="typing-dot"></span>
                                  <span className="typing-dot"></span>
                                </div>
                              ) : (
                                <div style={{ fontSize: '0.94rem' }}>{renderTextFormat(msg.text)}</div>
                              )}
                              
                              {/* RAG Source attributions */}
                              {msg.sources && msg.sources.length > 0 && (
                                <div style={{ marginTop: '10px' }}>
                                  <div style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', color: '#104891', marginBottom: '4px' }}>
                                    Source Evidence from Report:
                                  </div>
                                  {msg.sources.map((src, sIdx) => (
                                    <div key={sIdx} className="source-attribution">
                                      "{src}" (Source Page: {msg.metadata?.[sIdx]?.page || 1})
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          );
                        })}
                        <div ref={chatEndRef} />
                      </div>

                      <div className="chat-input-row">
                        <input
                          type="text"
                          className="chat-input"
                          placeholder="Ex: What does an elevated cholesterol mean for my diet?"
                          value={chatInput}
                          onChange={(e) => setChatInput(e.target.value)}
                          onKeyDown={(e) => e.key === 'Enter' && handleSendChat()}
                          disabled={isChatSending}
                        />
                        <button
                          className="primary-button"
                          onClick={handleSendChat}
                          disabled={isChatSending || !chatInput.trim()}
                        >
                          {isChatSending ? 'Searching...' : <Send size={16} />}
                        </button>
                      </div>
                    </div>

                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ============================================================ */}
        {/* COMMUNITY DASHBOARD VIEW                                     */}
        {/* ============================================================ */}
        {activeMode === 'community' && (
          <div>
            {/* Batch Upload Form */}
            <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginBottom: '24px' }}>
              <div className="metric-card" style={{ display: 'flex', flexDirection: 'column', gap: '8px', padding: '16px' }}>
                <label className="form-label">Clinic Location / Region</label>
                <select 
                  className="form-select"
                  value={selectedRegion}
                  onChange={(e) => setSelectedRegion(e.target.value)}
                >
                  <option value="Auto-assign (random)">Auto-assign (random)</option>
                  <option value="Urban-North">Urban-North</option>
                  <option value="Urban-South">Urban-South</option>
                  <option value="Rural-East">Rural-East</option>
                  <option value="Rural-West">Rural-West</option>
                </select>
              </div>

              <div className="metric-card" style={{ display: 'flex', flexDirection: 'column', gap: '8px', padding: '16px' }}>
                <label className="form-label">Anonymized Age Bracket</label>
                <select 
                  className="form-select"
                  value={selectedAgeGroup}
                  onChange={(e) => setSelectedAgeGroup(e.target.value)}
                >
                  <option value="Auto-assign (random)">Auto-assign (random)</option>
                  <option value="18-35">18-35</option>
                  <option value="36-59">36-59</option>
                  <option value="60+">60+</option>
                </select>
              </div>

              <div className="metric-card" style={{ padding: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <input 
                  type="file" 
                  accept=".pdf" 
                  multiple 
                  id="community-file-input"
                  style={{ display: 'none' }}
                  onChange={handleCommunityUpload}
                />
                <button 
                  className="primary-button" 
                  onClick={() => document.getElementById('community-file-input').click()}
                  style={{ width: '100%' }}
                >
                  <Upload size={16} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
                  Ingest Batch Reports
                </button>
              </div>
            </div>

            {loadingCommunity && (
              <div className="alert-card warning">
                <RefreshCw className="animate-spin" size={18} />
                <div>Loading public health datasets...</div>
              </div>
            )}

            {communityData && (
              <div>
                {/* Aggregate Population Metrics */}
                <div className="stats-grid">
                  <div className="metric-card">
                    <div className="value">{communityData.metrics.total_reports}</div>
                    <div className="label">Total Reports Logged</div>
                  </div>
                  <div className="metric-card">
                    <div className="value">{communityData.metrics.total_lab_values}</div>
                    <div className="label">Anonymized Test Parameters</div>
                  </div>
                  <div className="metric-card">
                    <div className="value" style={{ color: communityData.metrics.abnormal_rate > 25.0 ? '#e83a30' : '#16c79e' }}>
                      {communityData.metrics.abnormal_rate}%
                    </div>
                    <div className="label">Anomaly Rate</div>
                  </div>
                  <div className="metric-card">
                    <div className="value" style={{ color: '#104891' }}>
                      {useDP ? 'Epsilon 0.5' : 'Disabled'}
                    </div>
                    <div className="label">Privacy Guard</div>
                  </div>
                </div>

                {/* Alerts Section */}
                {communityData.alerts && communityData.alerts.length > 0 && (
                  <div style={{ marginBottom: '24px' }}>
                    <h3 style={{ margin: '0 0 12px 0' }}>🚨 Active Public Health Signals</h3>
                    {communityData.alerts.map((alert, idx) => (
                      <div key={idx} className={`alert-card ${alert.severity === 'critical' ? 'critical' : 'warning'}`}>
                        <span className="alert-icon-badge">
                          {alert.severity.toUpperCase()}
                        </span>
                        <div>
                          <strong>{alert.test_name.toUpperCase()} Spike</strong>: {alert.message}
                          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                            Action Recommended: {alert.action_recommendation}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Visualizations Panel */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '28px' }}>
                  <div className="metric-card" style={{ padding: '16px' }}>
                    {getFlagDistributionChart() && (
                      <PlotlyChart id="chart-flags" data={getFlagDistributionChart().data} layout={getFlagDistributionChart().layout} />
                    )}
                  </div>
                  
                  <div className="metric-card" style={{ padding: '16px' }}>
                    {getTopAbnormalChart() && (
                      <PlotlyChart id="chart-top-abnormal" data={getTopAbnormalChart().data} layout={getTopAbnormalChart().layout} />
                    )}
                  </div>
                </div>

                {/* Demographic Clustering Section */}
                <h3 style={{ margin: '24px 0 12px 0' }}>👥 Demographic Risk Clustering (Cross-Tab Analysis)</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px', marginBottom: '28px' }}>
                  <div className="metric-card" style={{ padding: '16px' }}>
                    {getHeatmapChart() && (
                      <PlotlyChart id="chart-heatmap" data={getHeatmapChart().data} layout={getHeatmapChart().layout} />
                    )}
                  </div>

                  <div className="metric-card" style={{ display: 'flex', flexDirection: 'column', justify: 'center' }}>
                    <h4 style={{ margin: '0 0 12px 0' }}>Demographic Observations</h4>
                    {communityData.demographic_highlights && communityData.demographic_highlights.length > 0 ? (
                      <ul style={{ paddingLeft: '20px', margin: 0, lineHeight: 1.6, fontSize: '0.88rem' }}>
                        {communityData.demographic_highlights.map((high, idx) => (
                          <li key={idx} style={{ marginBottom: '8px' }}>{high}</li>
                        ))}
                      </ul>
                    ) : (
                      <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', margin: 0 }}>
                        Insufficient cluster density to trace demographic anomalies. Ingest more records.
                      </p>
                    )}
                  </div>
                </div>

                {/* Seasonal Trends and Predictive Forecasting */}
                <h3 style={{ margin: '24px 0 12px 0' }}>📈 Epidemiological Trends & Least-Squares Forecasting</h3>
                <div className="metric-card" style={{ padding: '24px', marginBottom: '28px' }}>
                  <div style={{ display: 'flex', gap: '20px', marginBottom: '16px', alignItems: 'center' }}>
                    <div style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', gap: '6px' }}>
                      <label className="form-label">Target Test Parameter</label>
                      <select
                        className="form-select"
                        value={selectedTrendTest}
                        onChange={(e) => setSelectedTrendTest(e.target.value)}
                      >
                        {availableTests.map(t => (
                          <option key={t} value={t}>{t.toUpperCase()}</option>
                        ))}
                      </select>
                    </div>

                    <div style={{ width: '220px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                      <label className="form-label">Forecast Horizon: {trendDays} Days</label>
                      <input
                        type="range"
                        min="7"
                        max="90"
                        value={trendDays}
                        onChange={(e) => setTrendDays(parseInt(e.target.value))}
                        style={{ cursor: 'pointer' }}
                      />
                    </div>
                  </div>

                  {loadingTrend && (
                    <div style={{ textAlign: 'center', padding: '40px 0' }}>
                      <RefreshCw className="animate-spin" size={24} style={{ color: '#104891' }} />
                    </div>
                  )}

                  {!loadingTrend && getTrendChart() && (
                    <PlotlyChart id="chart-trends" data={getTrendChart().data} layout={getTrendChart().layout} />
                  )}
                </div>

                {/* Natural Language aggregates query */}
                <div className="chat-container">
                  <h3 style={{ marginTop: 0 }}>💬 Ask the Community Agent (Natural Language SQL Translator)</h3>
                  <p style={{ fontSize: '0.88rem', color: '#5e6b7c', margin: '0 0 16px 0' }}>
                    Formulate queries to parse counts, locations, and rates. The agent queries database schemas securely.
                  </p>
                  
                  <div className="chat-input-row" style={{ marginBottom: '16px' }}>
                    <input
                      type="text"
                      className="chat-input"
                      placeholder="Ex: Which age group had the highest abnormal cholesterol readings?"
                      value={communityQuestion}
                      onChange={(e) => setCommunityQuestion(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleCommunityChat()}
                      disabled={loadingCommChat}
                    />
                    <button
                      className="primary-button"
                      onClick={handleCommunityChat}
                      disabled={loadingCommChat || !communityQuestion.trim()}
                    >
                      {loadingCommChat ? 'Translating...' : 'Ask Database'}
                    </button>
                  </div>

                  {communityAnswer && (
                    <div 
                      className="metric-card" 
                      style={{ background: '#faf6f0', whiteSpace: 'pre-wrap', lineHeight: 1.6, fontSize: '0.94rem' }}
                    >
                      {communityAnswer}
                    </div>
                  )}
                </div>

              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
