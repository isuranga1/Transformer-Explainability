"use client";

import { useEffect, useState, useRef } from "react";
import { Stage, Layer, Image as KonvaImage, Line } from "react-konva";
import { Upload, Zap, RefreshCw, Sparkles, Activity } from "lucide-react";

export default function Page() {
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:5001";

  /* New State for UI Logic */
  const [perturbTestEnabled, setPerturbTestEnabled] = useState(false);
  const [modelId, setModelId] = useState("deit_tiny_patch16_224");
  const perturbSectionRef = useRef(null);

  /* Scroll to Perturbation Section when checkbox is checked, or top when unchecked */
  useEffect(() => {
    if (perturbTestEnabled && perturbSectionRef.current) {
      perturbSectionRef.current.scrollIntoView({ behavior: 'smooth' });
    } else if (!perturbTestEnabled) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, [perturbTestEnabled]);

  const [file, setFile] = useState(null);
  const [originalUrl, setOriginalUrl] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [attrMethod, setAttrMethod] = useState("transformer_attribution");
  const [device, setDevice] = useState("cpu");

  const [heatmapUrl, setHeatmapUrl] = useState(null);
  const [targetIndex, setTargetIndex] = useState("");
  const [initialPredictions, setInitialPredictions] = useState([]);
  const [originalPrediction, setOriginalPrediction] = useState(null);
  const [positiveResults, setPositiveResults] = useState([]);
  const [negativeResults, setNegativeResults] = useState([]);

  const [loadingHeatmap, setLoadingHeatmap] = useState(false);
  const [loadingPosPert, setLoadingPosPert] = useState(false);
  const [loadingNegPert, setLoadingNegPert] = useState(false);

  const [konvaImage, setKonvaImage] = useState(null);
  const [imgSize, setImgSize] = useState({ width: 0, height: 0 });
  const [maskLines, setMaskLines] = useState([]);
  const [isDrawingMask, setIsDrawingMask] = useState(false);

  const [perturbedUrl, setPerturbedUrl] = useState(null);
  const [inferenceResults, setInferenceResults] = useState([]);
  const [loadingInfer, setLoadingInfer] = useState(false);

  const stageWidth = 500;

  useEffect(() => {
    if (!originalUrl) {
      setKonvaImage(null);
      setImgSize({ width: 0, height: 0 });
      return;
    }
    const img = new window.Image();
    img.src = originalUrl;
    img.onload = () => {
      setKonvaImage(img);
      const aspect = img.height / img.width;
      setImgSize({
        width: stageWidth,
        height: stageWidth * aspect,
      });
    };
  }, [originalUrl]);

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0];
    if (!selected) return;

    setFile(selected);
    const url = URL.createObjectURL(selected);
    setOriginalUrl(url);

    setHeatmapUrl(null);
    setOriginalPrediction(null);
    setInitialPredictions([]);
    setPositiveResults([]);
    setNegativeResults([]);
    setMaskLines([]);
    setPerturbedUrl(null);
    setInferenceResults([]);
    setErrorMsg("");
  };

  const handleGenerateHeatmap = async () => {
    if (!file) {
      setErrorMsg("Please select an image first.");
      return;
    }

    setLoadingHeatmap(true);
    setErrorMsg("");
    setHeatmapUrl(null);

    try {
      const formData = new FormData();
      formData.append("image", file);
      if (targetIndex.trim() !== "") {
        formData.append("target_index", targetIndex.trim());
      }
      formData.append("method", attrMethod);
      formData.append("model_id", modelId);
      formData.append("device", device);

      const res = await fetch(`${API_BASE}/api/heatmap`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        let msg = `Heatmap request failed: ${res.status}`;
        try {
          const d = await res.json();
          if (d.error) msg = d.error;
        } catch (_) { }
        throw new Error(msg);
      }

      const blob = await res.blob();
      setHeatmapUrl(URL.createObjectURL(blob));

      /* Also fetch Top-5 Predictions for the Original Image */
      try {
        const formDataInfer = new FormData();
        formDataInfer.append("image", file);
        formDataInfer.append("model_id", modelId);
        formDataInfer.append("device", device);

        const resInfer = await fetch(`${API_BASE}/api/infer`, {
          method: "POST",
          body: formDataInfer,
        });

        if (resInfer.ok) {
          const dataInfer = await resInfer.json();
          setInitialPredictions(dataInfer.predictions || []);
        }
      } catch (e) {
        console.error("Failed to fetch initial predictions", e);
      }

    } catch (err) {
      console.error(err);
      setErrorMsg(err.message || "Something went wrong");
    } finally {
      setLoadingHeatmap(false);
    }
  };

  const runPerturbation = async (perturbationType) => {
    if (!file) {
      setErrorMsg("Please select an image first.");
      return;
    }

    if (perturbationType === "positive") setLoadingPosPert(true);
    if (perturbationType === "negative") setLoadingNegPert(true);

    setErrorMsg("");

    try {
      const formData = new FormData();
      formData.append("image", file);
      if (targetIndex.trim() !== "") {
        formData.append("target_index", targetIndex.trim());
      }
      formData.append("perturbation_type", perturbationType);
      formData.append("method", attrMethod);
      formData.append("model_id", modelId);
      formData.append("device", device);

      const res = await fetch(`${API_BASE}/api/perturbation`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        let msg = `Perturbation (${perturbationType}) failed: ${res.status}`;
        try {
          const d = await res.json();
          if (d.error) msg = d.error;
        } catch (_) { }
        throw new Error(msg);
      }

      const data = await res.json();
      setOriginalPrediction(data.original_prediction || null);

      if (perturbationType === "positive") {
        setPositiveResults(data.perturbation_results || []);
      } else {
        setNegativeResults(data.perturbation_results || []);
      }
    } catch (err) {
      console.error(err);
      setErrorMsg(err.message || "Something went wrong");
    } finally {
      if (perturbationType === "positive") setLoadingPosPert(false);
      if (perturbationType === "negative") setLoadingNegPert(false);
    }
  };

  const handlePositivePerturbation = () => runPerturbation("positive");
  const handleNegativePerturbation = () => runPerturbation("negative");

  const handleMaskMouseDown = (e) => {
    if (!konvaImage) return;
    const pos = e.target.getStage().getPointerPosition();
    setIsDrawingMask(true);
    setMaskLines((lines) => [
      ...lines,
      {
        points: [pos.x, pos.y],
      },
    ]);
  };

  const handleMaskMouseMove = (e) => {
    if (!isDrawingMask) return;
    const stage = e.target.getStage();
    const point = stage.getPointerPosition();
    setMaskLines((lines) => {
      const lastLine = lines[lines.length - 1];
      const newPoints = lastLine.points.concat([point.x, point.y]);
      const newLines = lines.slice(0, lines.length - 1);
      return [...newLines, { ...lastLine, points: newPoints }];
    });
  };

  const handleMaskMouseUp = () => {
    setIsDrawingMask(false);
  };

  const handleApplyCanvasPerturbation = () => {
    if (!konvaImage || maskLines.length === 0) {
      setErrorMsg("Draw a mask on the image first (paint over the region).");
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = imgSize.width;
    canvas.height = imgSize.height;
    const ctx = canvas.getContext("2d");

    ctx.drawImage(konvaImage, 0, 0, imgSize.width, imgSize.height);

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "black";
    ctx.lineWidth = 25;

    maskLines.forEach((line) => {
      const pts = line.points;
      if (pts.length < 4) return;
      ctx.beginPath();
      ctx.moveTo(pts[0], pts[1]);
      for (let i = 2; i < pts.length; i += 2) {
        ctx.lineTo(pts[i], pts[i + 1]);
      }
      ctx.stroke();
    });
    ctx.restore();

    const dataUrl = canvas.toDataURL("image/png");
    setPerturbedUrl(dataUrl);
    setInferenceResults([]);
    setErrorMsg("");
  };

  const handleRunInferencePerturbed = async () => {
    if (!perturbedUrl) {
      setErrorMsg("Apply a canvas perturbation first.");
      return;
    }

    setLoadingInfer(true);
    setErrorMsg("");
    setInferenceResults([]);

    try {
      const resBlob = await fetch(perturbedUrl);
      const blob = await resBlob.blob();

      const formData = new FormData();
      formData.append("image", blob, "perturbed.png");
      formData.append("device", device);

      const res = await fetch(`${API_BASE}/api/infer`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        let msg = `Inference failed: ${res.status}`;
        try {
          const d = await res.json();
          if (d.error) msg = d.error;
        } catch (_) { }
        throw new Error(msg);
      }

      const data = await res.json();
      setInferenceResults(data.predictions || []);
    } catch (err) {
      console.error(err);
      setErrorMsg(err.message || "Something went wrong");
    } finally {
      setLoadingInfer(false);
    }
  };

  return (
    <main className="min-h-screen relative" style={{ backgroundColor: '#0d1117', color: '#c9d1d9' }}>

      {/* 1. Fixed Header Bar */}
      <div
        className="fixed top-0 left-0 right-0 z-50 flex items-center gap-6 px-8 py-3 border-b shadow-md"
        style={{ backgroundColor: '#161b22', borderColor: '#30363d' }}
      >
        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="perturbTest"
            checked={perturbTestEnabled}
            onChange={(e) => setPerturbTestEnabled(e.target.checked)}
            className="w-5 h-5 rounded border-gray-600 bg-gray-700 text-blue-600 focus:ring-blue-500"
          />
          <label htmlFor="perturbTest" className="font-semibold select-none cursor-pointer">
            Perturb Test
          </label>
        </div>

        <div>
          <input
            type="file"
            id="header-file-upload"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
          />
          <label
            htmlFor="header-file-upload"
            className="cursor-pointer px-4 py-2 rounded-md font-semibold text-sm transition-colors"
            style={{ backgroundColor: '#238636', color: '#ffffff' }}
          >
            Upload Input Image
          </label>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold">model_id:</span>
          <select
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            style={{ backgroundColor: '#0d1117', borderColor: '#30363d' }}
            className="px-3 py-1.5 border rounded-md text-sm focus:ring-2 focus:ring-blue-500"
          >
            <option value="deit_tiny_patch16_224">deit_tiny_patch16_224</option>
            <option value="vit_base_patch16_224">vit_base_patch16_224</option>
            <option value="deit3_base_patch16_224">deit3_base_patch16_224</option>
          </select>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold">device:</span>
          <select
            value={device}
            onChange={(e) => setDevice(e.target.value)}
            style={{ backgroundColor: '#0d1117', borderColor: '#30363d' }}
            className="px-3 py-1.5 border rounded-md text-sm focus:ring-2 focus:ring-blue-500"
          >
            <option value="cpu">CPU</option>
            <option value="cuda">CUDA</option>
            <option value="mps">MPS</option>
          </select>
        </div>
      </div>

      {/* Spacer for fixed header */}
      <div className="h-20"></div>

      <div className="max-w-[1400px] mx-auto p-8 space-y-12">

        {/* 2. Restructured Main Content Area */}
        <div className="grid grid-cols-12 gap-8">
          {/* Left Control Column */}
          <div className="col-span-4 space-y-4">
            <div
              className="p-6 rounded-lg border space-y-4"
              style={{ backgroundColor: '#161b22', borderColor: '#30363d' }}
            >
              <div className="space-y-2">
                <label className="block text-sm font-semibold text-gray-400">
                  Attribution Method
                </label>
                <select
                  value={attrMethod}
                  onChange={(e) => setAttrMethod(e.target.value)}
                  style={{ backgroundColor: '#0d1117', borderColor: '#30363d' }}
                  className="w-full px-4 py-2.5 border rounded-lg focus:ring-2 focus:ring-blue-500"
                >
                  <option value="rollout">Rollout</option>
                  <option value="transformer_attribution">Transformer Attribution</option>
                  <option value="full">Full LRP</option>
                  <option value="last_layer">LRP Last Layer</option>
                  <option value="last_layer_attn">Attention Last Layer</option>
                  <option value="attn_gradcam">Attention GradCAM</option>
                </select>
              </div>

              <div className="space-y-2">
                <label className="block text-sm font-semibold text-gray-400">
                  Target Class Index (0-999)
                </label>
                <input
                  type="number"
                  min={0}
                  max={999}
                  value={targetIndex}
                  onChange={(e) => setTargetIndex(e.target.value)}
                  placeholder="Leave empty for top-1"
                  style={{ backgroundColor: '#0d1117', borderColor: '#30363d' }}
                  className="w-full px-4 py-2.5 border rounded-lg focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>

            <button
              onClick={handleGenerateHeatmap}
              disabled={loadingHeatmap || !file}
              style={{ backgroundColor: '#2f81f7', color: '#ffffff' }}
              className="w-full py-4 rounded-lg font-bold text-lg shadow hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              {loadingHeatmap ? "Generating..." : "Generate Explainability Map"}
            </button>
          </div>

          {/* Right Display Grid */}
          <div className="col-span-8 grid grid-cols-3 gap-4">
            {/* 1. Original Image Box */}
            <div
              className="aspect-square flex flex-col items-center justify-center p-2 rounded-lg border text-center"
              style={{ backgroundColor: '#161b22', borderColor: '#30363d' }}
            >
              {originalUrl ? (
                <img src={originalUrl} alt="Original" className="w-full h-full object-contain rounded" />
              ) : (
                <span className="text-gray-500">Original Uploaded Image<br />Should Be viewed here</span>
              )}
            </div>

            {/* 2. Generated Heatmap Box */}
            <div
              className="aspect-square flex flex-col items-center justify-center p-2 rounded-lg border text-center"
              style={{ backgroundColor: '#161b22', borderColor: '#30363d' }}
            >
              {heatmapUrl ? (
                <div className="w-full h-full relative overflow-hidden flex items-center justify-center">
                  <img src={heatmapUrl} alt="Heatmap" className="w-full h-full object-contain rounded" />
                </div>
              ) : (
                <span className="text-gray-500">Generated Heatmap<br />should be viewed here</span>
              )}
            </div>

            {/* 3. Top-5 Predictions Box */}
            <div
              className="aspect-square flex flex-col p-4 rounded-lg border overflow-y-auto"
              style={{ backgroundColor: '#161b22', borderColor: '#30363d' }}
            >
              <h3 className="font-bold text-lg mb-3 sticky top-0 bg-[#161b22] pb-2 border-b border-gray-700">Top-5 Predictions</h3>
              {initialPredictions.length > 0 ? (
                <div className="space-y-2 text-sm">
                  {initialPredictions.map((pred, i) => (
                    <div key={i} className="flex justify-between items-center p-2 rounded hover:bg-white/5">
                      <div className="flex flex-col">
                        <span className="font-semibold text-blue-400">{pred.class_name}</span>
                        <span className="text-xs text-gray-500">ID: {pred.class_idx}</span>
                      </div>
                      <span className="font-mono text-green-500 font-bold">{(pred.prob * 100).toFixed(2)}%</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex-1 flex flex-col items-center justify-center text-gray-500 text-center">
                  <Activity className="w-8 h-8 mb-2 opacity-50" />
                  <span>Predictions will appear here after generation</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* 3. Layerwise Raw Attention Propagation Placeholder */}
        <div
          className="w-full h-64 flex items-center justify-center rounded-lg border-2 border-dashed"
          style={{ backgroundColor: '#0d1117', borderColor: '#30363d' }}
        >
          <h2 className="text-3xl font-light text-center text-gray-500">
            Layerwise Raw Attention Propogation to be implemented later
          </h2>
        </div>

        {/* 4. Horizontal Line */}
        <hr className="border-t-2" style={{ borderColor: '#8b949e' }} />

        {/* 5. Perturb Test Section */}
        <div ref={perturbSectionRef} className="space-y-6 pt-4">
          <h2 className="text-4xl font-normal text-center mb-10" style={{ color: '#c9d1d9' }}>
            Content from Perturb test (Dont change this , keep it as it is)
          </h2>

          {/* Copy existing perturbation UI logic here */}
          <div style={{ backgroundColor: '#161b22', borderColor: '#30363d' }} className="rounded-lg border p-6 space-y-6">
            <div className="flex flex-wrap gap-3 mb-6">
              <button
                onClick={handlePositivePerturbation}
                disabled={loadingPosPert || !file}
                style={{
                  backgroundColor: loadingPosPert || !file ? '#21262d' : '#238636',
                  borderColor: '#30363d',
                  color: '#ffffff'
                }}
                className="flex items-center gap-2 px-6 py-3 border rounded-lg font-semibold hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
              >
                <Zap className="w-4 h-4" />
                {loadingPosPert ? "Running..." : "Positive Perturbation"}
              </button>

              <button
                onClick={handleNegativePerturbation}
                disabled={loadingNegPert || !file}
                style={{
                  backgroundColor: loadingNegPert || !file ? '#21262d' : '#da3633',
                  borderColor: '#30363d',
                  color: '#ffffff'
                }}
                className="flex items-center gap-2 px-6 py-3 border rounded-lg font-semibold hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
              >
                <RefreshCw className="w-4 h-4" />
                {loadingNegPert ? "Running..." : "Negative Perturbation"}
              </button>
              {errorMsg && (
                <span className="text-red-500 ml-4 self-center">{errorMsg}</span>
              )}
            </div>

            {(originalPrediction ||
              positiveResults.length > 0 ||
              negativeResults.length > 0) && (
                <div className="space-y-6">
                  {/* Reusing existing perturbation visualization code */}
                  {originalPrediction && (
                    <div style={{ backgroundColor: '#1c2128', borderColor: '#30363d' }} className="rounded-lg p-4 space-y-2 border">
                      <div className="flex items-baseline gap-2">
                        <span className="font-semibold">Top-1 (Original):</span>
                        <span>{originalPrediction.top1.class_name}</span>
                        <span className="text-sm text-gray-500">(idx {originalPrediction.top1.class_idx})</span>
                        <span className="ml-auto font-bold text-blue-400">{(originalPrediction.top1.prob * 100).toFixed(2)}%</span>
                      </div>
                      <div className="flex items-baseline gap-2">
                        <span className="font-semibold">Target Class:</span>
                        <span>{originalPrediction.target.class_name}</span>
                        <span className="text-sm text-gray-500">(idx {originalPrediction.target.class_idx})</span>
                        <span className="ml-auto font-bold text-blue-400">{(originalPrediction.target.prob * 100).toFixed(2)}%</span>
                      </div>
                    </div>
                  )}

                  {positiveResults.length > 0 && (
                    <div className="space-y-3">
                      <h4 className="text-lg font-semibold text-green-500">Positive Perturbation Results</h4>
                      <div className="overflow-x-auto rounded-lg border border-gray-700">
                        <table className="w-full">
                          <thead className="bg-[#161b22]">
                            <tr>
                              <th className="px-4 py-2 text-left">Fraction</th>
                              <th className="px-4 py-2 text-left">Top-1 Class</th>
                              <th className="px-4 py-2 text-left">Top-1 Prob</th>
                              <th className="px-4 py-2 text-left">Target Prob</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-gray-700">
                            {positiveResults.map((r, i) => (
                              <tr key={i}>
                                <td className="px-4 py-2">{(r.fraction * 100).toFixed(0)}%</td>
                                <td className="px-4 py-2">{r.top1_class_name}</td>
                                <td className="px-4 py-2">{(r.top1_prob * 100).toFixed(2)}%</td>
                                <td className="px-4 py-2">{(r.target_prob * 100).toFixed(2)}%</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {negativeResults.length > 0 && (
                    <div className="space-y-3">
                      <h4 className="text-lg font-semibold text-red-500">Negative Perturbation Results</h4>
                      <div className="overflow-x-auto rounded-lg border border-gray-700">
                        <table className="w-full">
                          <thead className="bg-[#161b22]">
                            <tr>
                              <th className="px-4 py-2 text-left">Fraction</th>
                              <th className="px-4 py-2 text-left">Top-1 Class</th>
                              <th className="px-4 py-2 text-left">Top-1 Prob</th>
                              <th className="px-4 py-2 text-left">Target Prob</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-gray-700">
                            {negativeResults.map((r, i) => (
                              <tr key={i}>
                                <td className="px-4 py-2">{(r.fraction * 100).toFixed(0)}%</td>
                                <td className="px-4 py-2">{r.top1_class_name}</td>
                                <td className="px-4 py-2">{(r.top1_prob * 100).toFixed(2)}%</td>
                                <td className="px-4 py-2">{(r.target_prob * 100).toFixed(2)}%</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              )}
          </div>

          {/* Mask-Based Perturbation - Keeping as is */}
          <div className="grid lg:grid-cols-2 gap-6">
            <div style={{ backgroundColor: '#161b22', borderColor: '#30363d' }} className="rounded-lg border p-6 space-y-4">
              <h3 className="text-xl font-bold">Mask-based Perturbation</h3>
              <div style={{ borderColor: '#30363d', backgroundColor: '#0d1117' }} className="rounded-lg overflow-hidden border-2 border-dashed">
                {konvaImage && imgSize.width > 0 && imgSize.height > 0 ? (
                  <Stage
                    width={imgSize.width}
                    height={imgSize.height}
                    onMouseDown={handleMaskMouseDown}
                    onMouseMove={handleMaskMouseMove}
                    onMouseUp={handleMaskMouseUp}
                    style={{ cursor: "crosshair" }}
                  >
                    <Layer>
                      <KonvaImage
                        image={konvaImage}
                        x={0}
                        y={0}
                        width={imgSize.width}
                        height={imgSize.height}
                      />
                    </Layer>
                    <Layer>
                      {maskLines.map((line, idx) => (
                        <Line
                          key={idx}
                          points={line.points}
                          stroke="#ef4444"
                          strokeWidth={20}
                          tension={0.5}
                          lineCap="round"
                          lineJoin="round"
                          opacity={0.7}
                        />
                      ))}
                    </Layer>
                  </Stage>
                ) : (
                  <div className="p-16 text-center text-gray-500">
                    <Upload className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>Upload an image to start drawing</p>
                  </div>
                )}
              </div>
              <div className="flex gap-3">
                <button
                  onClick={handleApplyCanvasPerturbation}
                  disabled={!originalUrl || maskLines.length === 0}
                  style={{ backgroundColor: '#238636', color: '#ffffff' }}
                  className="flex-1 py-3 rounded-lg font-semibold disabled:opacity-50"
                >
                  Apply Mask
                </button>
                <button
                  onClick={handleRunInferencePerturbed}
                  disabled={!perturbedUrl || loadingInfer}
                  style={{ backgroundColor: '#238636', color: '#ffffff' }}
                  className="flex-1 py-3 rounded-lg font-semibold disabled:opacity-50"
                >
                  Run Inference
                </button>
              </div>
            </div>

            {/* Results Preview */}
            <div style={{ backgroundColor: '#161b22', borderColor: '#30363d' }} className="rounded-lg border p-6 space-y-4">
              <h3 className="text-xl font-bold">Preview & Results</h3>
              <div className="grid grid-cols-2 gap-4">
                {perturbedUrl && (
                  <div>
                    <p className="text-sm font-semibold mb-2">Perturbed</p>
                    <img src={perturbedUrl} alt="perturbed" className="w-full rounded border border-gray-700" />
                  </div>
                )}
              </div>
              {inferenceResults.length > 0 && (
                <div className="space-y-2">
                  {inferenceResults.map((r, idx) => (
                    <div key={idx} className="flex justify-between bg-[#0d1117] p-2 rounded border border-gray-700">
                      <span>{idx + 1}. {r.class_name}</span>
                      <span className="text-blue-400 font-bold">{(r.prob * 100).toFixed(2)}%</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

      </div>
    </main>
  );
}
