import { useState } from "react";
import axios from "axios";

const API_URL = "http://127.0.0.1:8000";

interface LimeWord {
  word: string;
  contribution: number;
}

interface Result {
  verdict: string;
  confidence: number;
  language: string;
  red_flags: string[];
  explanation: string;
  fact_check_sources: string[];
  lime_explanation: LimeWord[];
  layer_scores: {
    xlm_roberta: number;
    groq_llama_70b: number;
    fact_check: number;
  };
}

export default function App() {
  const [text, setText]       = useState("");
  const [url, setUrl]         = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState<Result | null>(null);
  const [error, setError]     = useState("");
  const [mode, setMode]       = useState<"text" | "url">("text");

  const analyze = async () => {
    if (mode === "text" && !text.trim()) return;
    if (mode === "url"  && !url.trim())  return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const endpoint = mode === "text" ? "/analyze/text" : "/analyze/url";
      const body     = mode === "text" ? { text } : { url };
      const res      = await axios.post(`${API_URL}${endpoint}`, body);
      setResult(res.data);
    } catch (e) {
      setError("Failed to connect to API. Make sure the backend is running.");
    }
    setLoading(false);
  };

  const isFake = result?.verdict === "FAKE";

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <div className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center gap-3">
          <div className="text-2xl">🔍</div>
          <div>
            <h1 className="text-xl font-bold text-white">Multilingual Fake News Detector</h1>
            <p className="text-gray-400 text-sm">XLM-RoBERTa + Groq Llama-70B + Google Fact Check</p>
          </div>
          <div className="ml-auto flex gap-2">
            <span className="bg-green-900 text-green-300 text-xs px-2 py-1 rounded-full">96.53% F1</span>
            <span className="bg-blue-900 text-blue-300 text-xs px-2 py-1 rounded-full">100+ Languages</span>
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-6 py-8">
        {/* Input Card */}
        <div className="bg-gray-900 rounded-2xl border border-gray-800 p-6 mb-6">
          {/* Mode Toggle */}
          <div className="flex gap-2 mb-4">
            <button
              onClick={() => setMode("text")}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                mode === "text"
                  ? "bg-blue-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:bg-gray-700"
              }`}
            >
              📝 Paste Text
            </button>
            <button
              onClick={() => setMode("url")}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                mode === "url"
                  ? "bg-blue-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:bg-gray-700"
              }`}
            >
              🔗 Enter URL
            </button>
          </div>

          {mode === "text" ? (
            <textarea
              className="w-full bg-gray-800 border border-gray-700 rounded-xl p-4 text-white placeholder-gray-500 resize-none focus:outline-none focus:border-blue-500 transition-colors"
              rows={6}
              placeholder="Paste your news article here in any language — English, Hindi, Tamil, Telugu, Kannada, Bengali..."
              value={text}
              onChange={e => setText(e.target.value)}
            />
          ) : (
            <input
              className="w-full bg-gray-800 border border-gray-700 rounded-xl p-4 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 transition-colors"
              placeholder="Enter news article URL..."
              value={url}
              onChange={e => setUrl(e.target.value)}
            />
          )}

          <button
            onClick={analyze}
            disabled={loading}
            className="mt-4 w-full bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-xl transition-all"
          >
            {loading ? "🔄 Analyzing with 3 AI layers..." : "🔍 Detect Fake News"}
          </button>

          {error && (
            <div className="mt-3 bg-red-900/50 border border-red-700 text-red-300 px-4 py-3 rounded-lg text-sm">
              {error}
            </div>
          )}
        </div>

        {/* Results */}
        {result && (
          <div className="space-y-4">
            {/* Verdict Banner */}
            <div className={`rounded-2xl border p-6 ${
              isFake
                ? "bg-red-950 border-red-800"
                : "bg-green-950 border-green-800"
            }`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="text-5xl">{isFake ? "🚨" : "✅"}</div>
                  <div>
                    <div className={`text-3xl font-black ${isFake ? "text-red-400" : "text-green-400"}`}>
                      {result.verdict}
                    </div>
                    <div className="text-gray-300 text-sm mt-1">
                      {result.confidence}% confidence • Language: {result.language.toUpperCase()}
                    </div>
                  </div>
                </div>
                {/* Confidence Circle */}
                <div className={`w-20 h-20 rounded-full border-4 flex items-center justify-center ${
                  isFake ? "border-red-500" : "border-green-500"
                }`}>
                  <span className="text-xl font-bold">{result.confidence}%</span>
                </div>
              </div>

              {/* Confidence Bar */}
              <div className="mt-4">
                <div className="bg-gray-800 rounded-full h-3">
                  <div
                    className={`h-3 rounded-full transition-all ${isFake ? "bg-red-500" : "bg-green-500"}`}
                    style={{ width: `${result.confidence}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Layer Scores */}
            <div className="bg-gray-900 rounded-2xl border border-gray-800 p-6">
              <h3 className="font-bold text-gray-200 mb-4">🧠 3-Layer AI Analysis</h3>
              <div className="space-y-3">
                {[
                  { name: "XLM-RoBERTa (Fine-tuned)", score: result.layer_scores.xlm_roberta, color: "blue" },
                  { name: "Groq Llama-70B", score: result.layer_scores.groq_llama_70b, color: "purple" },
                  { name: "Google Fact Check", score: result.layer_scores.fact_check, color: "orange" },
                ].map((layer) => (
                  <div key={layer.name}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-300">{layer.name}</span>
                      <span className={`font-bold ${
                        layer.score > 50 ? "text-red-400" : "text-green-400"
                      }`}>
                        {layer.score > 50 ? `${layer.score}% FAKE` : `${100 - layer.score}% REAL`}
                      </span>
                    </div>
                    <div className="bg-gray-800 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${layer.score > 50 ? "bg-red-500" : "bg-green-500"}`}
                        style={{ width: `${layer.score > 50 ? layer.score : 100 - layer.score}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* LIME Explanation */}
            {result.lime_explanation.length > 0 && (
              <div className="bg-gray-900 rounded-2xl border border-gray-800 p-6">
                <h3 className="font-bold text-gray-200 mb-1">🔬 LIME Word Analysis</h3>
                <p className="text-gray-500 text-xs mb-4">Words that most influenced the AI decision</p>
                <div className="flex flex-wrap gap-2">
                  {result.lime_explanation.map((item, i) => (
                    <span
                      key={i}
                      className={`px-3 py-1 rounded-full text-sm font-medium border ${
                        item.contribution > 0
                          ? "bg-red-900/50 border-red-700 text-red-300"
                          : "bg-green-900/50 border-green-700 text-green-300"
                      }`}
                    >
                      {item.word}
                      <span className="ml-1 opacity-70 text-xs">
                        {item.contribution > 0 ? "+" : ""}{item.contribution.toFixed(3)}
                      </span>
                    </span>
                  ))}
                </div>
                <p className="text-gray-600 text-xs mt-3">
                  🔴 Red = contributed to FAKE verdict &nbsp;|&nbsp; 🟢 Green = contributed to REAL verdict
                </p>
              </div>
            )}

            {/* Red Flags */}
            {result.red_flags.length > 0 && (
              <div className="bg-gray-900 rounded-2xl border border-gray-800 p-6">
                <h3 className="font-bold text-gray-200 mb-3">🚩 Red Flags Detected</h3>
                <div className="flex flex-wrap gap-2">
                  {result.red_flags.map((flag, i) => (
                    <span key={i} className="bg-red-900/50 border border-red-700 text-red-300 px-3 py-1 rounded-full text-sm">
                      {flag}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Explanation */}
            <div className="bg-gray-900 rounded-2xl border border-gray-800 p-6">
              <h3 className="font-bold text-gray-200 mb-2">💡 AI Reasoning</h3>
              <p className="text-gray-300 text-sm leading-relaxed">{result.explanation}</p>
            </div>

            {/* Fact Check Sources */}
            {result.fact_check_sources.length > 0 && (
              <div className="bg-gray-900 rounded-2xl border border-gray-800 p-6">
                <h3 className="font-bold text-gray-200 mb-3">📋 Fact Check Sources</h3>
                <ul className="space-y-2">
                  {result.fact_check_sources.map((source, i) => (
                    <li key={i} className="text-gray-300 text-sm flex items-start gap-2">
                      <span className="text-blue-400 mt-0.5">•</span>
                      {source}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}