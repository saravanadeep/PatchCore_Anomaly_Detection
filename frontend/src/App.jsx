import { useEffect, useState } from "react";
import "./inspection.css";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [opacity, setOpacity] = useState(100);

  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function selectFile(event) {
    const next = event.target.files?.[0];
    if (!next) return;
    setFile(next);
    setResult(null);
    setError("");
    setOpacity(100);
  }

  async function inspect() {
    if (!file || busy) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const response = await fetch("/predict", { method: "POST", body });
      const data = await response.json().catch(() => null);
      if (!response.ok || !data) {
        throw new Error(typeof data?.detail === "string" ? data.detail : "Inspection unavailable. Check that FastAPI is running, then retry.");
      }
      if (!Number.isFinite(data.score) || !Number.isFinite(data.threshold) ||
          !["Normal", "Defective"].includes(data.prediction) || !data.heatmap?.startsWith("data:image/png;base64,")) {
        throw new Error("The API returned an incomplete inspection result.");
      }
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return <div className="inspection">
    <header className="mast"><div><b className="brand"><i aria-hidden="true"/>PatchCore</b><span>Bottle inspection</span></div></header>
    <main>
      <h1>Inspection bench</h1>
      <div className="toolbar">
        <label className="pick">Choose image<input type="file" accept="image/png,image/jpeg" disabled={busy} onChange={selectFile}/></label>
        <span className="filename">{file?.name || "Select a bottle photo · PNG or JPEG"}</span>
        <button className="run" disabled={!file || busy} onClick={inspect}>{busy ? "Inspecting…" : "Inspect bottle →"}</button>
      </div>
      {error && <p className="err" role="alert">{error}</p>}
      <section className="summary" aria-label="Inspection result" aria-live="polite">
        <div><small>Prediction</small><strong className={result ? result.prediction : ""}>{result?.prediction || (busy ? "Inspecting…" : "Awaiting inspection")}</strong></div>
        <div><small>Anomaly score</small><strong>{result ? result.score.toFixed(3) : "—"}</strong></div>
        <div><small>Decision cutoff</small><strong>{result ? result.threshold.toFixed(3) : "—"}</strong></div>
      </section>
      <div className="panels">
        <section><h2>Original image</h2><div className="stage">
          {preview ? <img src={preview} alt="Uploaded bottle"/> : <span>Choose an image to preview it</span>}
        </div></section>
        <section><h2>Relative anomaly heatmap</h2><div className="stage">
          {result ? <div className="layers"><img src={preview} alt="Bottle with adjustable heatmap overlay"/><img style={{opacity: opacity / 100}} src={result.heatmap} alt=""/></div> : <span>{busy ? "Calculating patch distances…" : "Run inspection to see the heatmap"}</span>}
        </div></section>
      </div>
      {result && <div className="controls"><label>Overlay strength <input type="range" min="0" max="100" value={opacity} onChange={e => setOpacity(Number(e.target.value))}/><output>{opacity}%</output></label>
        <p>Red marks the most unusual regions within this image. A score above the cutoff flags the bottle as defective.</p></div>}
    </main>
  </div>;
}
