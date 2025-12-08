"use client";

import { useEffect, useState } from "react";
import { Stage, Layer, Image as KonvaImage, Line } from "react-konva";
import { Upload, Zap, RefreshCw, Sparkles, Activity } from "lucide-react";

export default function Page() {
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:5001";

  const [file, setFile] = useState(null);
  const [originalUrl, setOriginalUrl] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [attrMethod, setAttrMethod] = useState("transformer_attribution");
  const [device, setDevice] = useState("cpu");

  const [heatmapUrl, setHeatmapUrl] = useState(null);
  const [targetIndex, setTargetIndex] = useState("");
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
        } catch (_) {}
        throw new Error(msg);
      }

      const blob = await res.blob();
      setHeatmapUrl(URL.createObjectURL(blob));
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
        } catch (_) {}
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
        } catch (_) {}
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
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center space-y-3">
          <div className="flex items-center justify-center gap-3">
            <Activity className="w-10 h-10 text-indigo-400" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-indigo-400 to-blue-400 bg-clip-text text-transparent">
              ViT-LRP Explorer
            </h1>
          </div>
          <p className="text-slate-400 text-lg">
            Analyze model predictions with heatmaps, LRP perturbations &
            mask-based occlusion
          </p>
        </div>

        {/* Control Panel */}
        <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl shadow-xl border border-slate-700 p-6 space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            {/* File Upload */}
            <div className="space-y-2">
              <label className="block text-sm font-semibold text-slate-300">
                Upload Image
              </label>
              <div className="relative">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="block w-full text-sm text-slate-300 file:mr-4 file:py-2.5 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-indigo-900 file:text-indigo-300 hover:file:bg-indigo-800 file:cursor-pointer cursor-pointer border border-slate-600 bg-slate-900 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
            </div>

            {/* Target Index */}
            <div className="space-y-2">
              <label className="block text-sm font-semibold text-slate-700">
                Target Class Index (0-999)
              </label>
              <input
                type="number"
                min={0}
                max={999}
                value={targetIndex}
                onChange={(e) => setTargetIndex(e.target.value)}
                placeholder="Leave empty for top-1"
                className="w-full px-4 py-2.5 border border-slate-600 bg-slate-900 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-slate-200 placeholder:text-slate-500"
              />
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {/* Attribution Method */}
            <div className="space-y-2">
              <label className="block text-sm font-semibold text-slate-300">
                Attribution Method
              </label>
              <select
                value={attrMethod}
                onChange={(e) => setAttrMethod(e.target.value)}
                className="w-full px-4 py-2.5 border border-slate-600 bg-slate-900 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-slate-200"
              >
                <option value="rollout">Rollout</option>
                <option value="transformer_attribution">
                  Transformer Attribution
                </option>
                <option value="full">Full LRP</option>
                <option value="last_layer">LRP Last Layer</option>
                <option value="last_layer_attn">Attention Last Layer</option>
                <option value="attn_gradcam">Attention GradCAM</option>
              </select>
            </div>

            {/* Device Selection */}
            <div className="space-y-2">
              <label className="block text-sm font-semibold text-slate-300">
                Device
              </label>
              <select
                value={device}
                onChange={(e) => setDevice(e.target.value)}
                className="w-full px-4 py-2.5 border border-slate-600 bg-slate-900 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-slate-200"
              >
                <option value="cpu">CPU</option>
                <option value="cuda">CUDA (GPU)</option>
                <option value="mps">MPS (Apple Silicon)</option>
              </select>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-wrap gap-3">
            <button
              onClick={handleGenerateHeatmap}
              disabled={loadingHeatmap || !file}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-700 hover:to-indigo-800 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
            >
              <Sparkles className="w-4 h-4" />
              {loadingHeatmap ? "Generating..." : "Generate Heatmap"}
            </button>

            <button
              onClick={handlePositivePerturbation}
              disabled={loadingPosPert || !file}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-emerald-600 to-emerald-700 hover:from-emerald-700 hover:to-emerald-800 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
            >
              <Zap className="w-4 h-4" />
              {loadingPosPert ? "Running..." : "Positive Perturbation"}
            </button>

            <button
              onClick={handleNegativePerturbation}
              disabled={loadingNegPert || !file}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-rose-600 to-rose-700 hover:from-rose-700 hover:to-rose-800 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
            >
              <RefreshCw className="w-4 h-4" />
              {loadingNegPert ? "Running..." : "Negative Perturbation"}
            </button>
          </div>

          {errorMsg && (
            <div className="bg-rose-900/30 border border-rose-800 text-rose-300 px-4 py-3 rounded-lg">
              {errorMsg}
            </div>
          )}
        </div>

        {/* Image Display */}
        {(originalUrl || heatmapUrl) && (
          <div className="grid md:grid-cols-2 gap-6">
            {originalUrl && (
              <div className="bg-slate-800 rounded-2xl shadow-xl border border-slate-700 p-6 space-y-4">
                <div className="flex items-center gap-2 text-slate-200 font-semibold">
                  <Activity className="w-5 h-5" />
                  <h3 className="text-lg">Original Image</h3>
                </div>
                <img
                  src={originalUrl}
                  alt="original"
                  className="w-full h-auto rounded-xl border border-slate-700"
                />
              </div>
            )}

            {heatmapUrl && (
              <div className="bg-white rounded-2xl shadow-xl border border-slate-200 p-6 space-y-4">
                <div className="flex items-center gap-2 text-slate-700 font-semibold">
                  <Activity className="w-5 h-5" />
                  <h3 className="text-lg">LRP Heatmap</h3>
                </div>
                <img
                  src={heatmapUrl}
                  alt="heatmap"
                  className="w-full h-auto rounded-xl border border-slate-200"
                />
              </div>
            )}
          </div>
        )}

        {/* Perturbation Analysis */}
        {(originalPrediction ||
          positiveResults.length > 0 ||
          negativeResults.length > 0) && (
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-2xl shadow-xl border border-slate-700 p-6 space-y-6">
            <h3 className="text-2xl font-bold text-slate-100">
              LRP-based Perturbation Analysis
            </h3>

            {originalPrediction && (
              <div className="bg-gradient-to-r from-blue-900/30 to-indigo-900/30 rounded-xl p-4 space-y-2 border border-slate-700">
                <div className="flex items-baseline gap-2">
                  <span className="font-semibold text-slate-300">
                    Top-1 (Original):
                  </span>
                  <span className="text-slate-100">
                    {originalPrediction.top1.class_name}
                  </span>
                  <span className="text-slate-400 text-sm">
                    (idx {originalPrediction.top1.class_idx})
                  </span>
                  <span className="ml-auto font-bold text-indigo-400">
                    {(originalPrediction.top1.prob * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="flex items-baseline gap-2">
                  <span className="font-semibold text-slate-300">
                    Target Class:
                  </span>
                  <span className="text-slate-100">
                    {originalPrediction.target.class_name}
                  </span>
                  <span className="text-slate-400 text-sm">
                    (idx {originalPrediction.target.class_idx})
                  </span>
                  <span className="ml-auto font-bold text-indigo-400">
                    {(originalPrediction.target.prob * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            )}

            {positiveResults.length > 0 && (
              <div className="space-y-3">
                <h4 className="text-lg font-semibold text-emerald-400">
                  Positive Perturbation (mask high-importance pixels)
                </h4>
                <div className="overflow-x-auto rounded-xl border border-slate-700">
                  <table className="w-full">
                    <thead className="bg-emerald-900/30">
                      <tr>
                        <th className="px-4 py-3 text-left text-sm font-semibold text-slate-300">
                          Fraction
                        </th>
                        <th className="px-4 py-3 text-left text-sm font-semibold text-slate-300">
                          Top-1 Class
                        </th>
                        <th className="px-4 py-3 text-left text-sm font-semibold text-slate-300">
                          Top-1 Prob
                        </th>
                        <th className="px-4 py-3 text-left text-sm font-semibold text-slate-300">
                          Target Prob
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-700">
                      {positiveResults.map((r) => (
                        <tr
                          key={`pos-${r.fraction}`}
                          className="hover:bg-slate-700/30"
                        >
                          <td className="px-4 py-3 text-slate-300">
                            {(r.fraction * 100).toFixed(0)}%
                          </td>
                          <td className="px-4 py-3 text-slate-300">
                            {r.top1_class_name}{" "}
                            <span className="text-slate-500 text-sm">
                              (idx {r.top1_class_idx})
                            </span>
                          </td>
                          <td className="px-4 py-3 font-semibold text-slate-200">
                            {(r.top1_prob * 100).toFixed(2)}%
                          </td>
                          <td className="px-4 py-3 font-semibold text-slate-200">
                            {(r.target_prob * 100).toFixed(2)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {negativeResults.length > 0 && (
              <div className="space-y-3">
                <h4 className="text-lg font-semibold text-rose-400">
                  Negative Perturbation (mask low-importance pixels)
                </h4>
                <div className="overflow-x-auto rounded-xl border border-slate-700">
                  <table className="w-full">
                    <thead className="bg-rose-900/30">
                      <tr>
                        <th className="px-4 py-3 text-left text-sm font-semibold text-slate-300">
                          Fraction
                        </th>
                        <th className="px-4 py-3 text-left text-sm font-semibold text-slate-300">
                          Top-1 Class
                        </th>
                        <th className="px-4 py-3 text-left text-sm font-semibold text-slate-300">
                          Top-1 Prob
                        </th>
                        <th className="px-4 py-3 text-left text-sm font-semibold text-slate-300">
                          Target Prob
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-700">
                      {negativeResults.map((r) => (
                        <tr
                          key={`neg-${r.fraction}`}
                          className="hover:bg-slate-700/30"
                        >
                          <td className="px-4 py-3 text-slate-300">
                            {(r.fraction * 100).toFixed(0)}%
                          </td>
                          <td className="px-4 py-3 text-slate-300">
                            {r.top1_class_name}{" "}
                            <span className="text-slate-500 text-sm">
                              (idx {r.top1_class_idx})
                            </span>
                          </td>
                          <td className="px-4 py-3 font-semibold text-slate-200">
                            {(r.top1_prob * 100).toFixed(2)}%
                          </td>
                          <td className="px-4 py-3 font-semibold text-slate-200">
                            {(r.target_prob * 100).toFixed(2)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Mask-based Perturbation */}
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Drawing Canvas */}
          <div className="bg-slate-800 rounded-2xl shadow-xl border border-slate-700 p-6 space-y-4">
            <h3 className="text-xl font-bold text-slate-100">
              Mask-based Perturbation
            </h3>
            <p className="text-sm text-slate-400">
              Draw over the image to create a segmentation mask. The painted
              region will be occluded for inference.
            </p>

            <div className="rounded-xl overflow-hidden border-2 border-dashed border-slate-600 bg-slate-900">
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
                <div className="p-16 text-center text-slate-500">
                  <Upload className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>Upload an image to start drawing</p>
                </div>
              )}
            </div>

            <div className="flex gap-3">
              <button
                onClick={handleApplyCanvasPerturbation}
                disabled={!originalUrl || maskLines.length === 0}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
              >
                <Sparkles className="w-4 h-4" />
                Apply Mask
              </button>

              <button
                onClick={handleRunInferencePerturbed}
                disabled={!perturbedUrl || loadingInfer}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-emerald-600 to-emerald-700 hover:from-emerald-700 hover:to-emerald-800 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
              >
                <Activity className="w-4 h-4" />
                {loadingInfer ? "Running..." : "Run Inference"}
              </button>
            </div>
          </div>

          {/* Results */}
          <div className="bg-slate-800 rounded-2xl shadow-xl border border-slate-700 p-6 space-y-4">
            <h3 className="text-xl font-bold text-slate-100">
              Preview & Results
            </h3>

            <div className="grid grid-cols-2 gap-4">
              {originalUrl && (
                <div className="space-y-2">
                  <p className="text-sm font-semibold text-slate-400">
                    Original
                  </p>
                  <img
                    src={originalUrl}
                    alt="original-preview"
                    className="w-full h-auto rounded-lg border border-slate-700"
                  />
                </div>
              )}
              {perturbedUrl && (
                <div className="space-y-2">
                  <p className="text-sm font-semibold text-slate-400">
                    Perturbed
                  </p>
                  <img
                    src={perturbedUrl}
                    alt="perturbed-preview"
                    className="w-full h-auto rounded-lg border border-slate-200"
                  />
                </div>
              )}
            </div>

            {inferenceResults.length > 0 && (
              <div className="bg-gradient-to-r from-blue-900/30 to-indigo-900/30 rounded-xl p-4 space-y-3 border border-slate-700">
                <h4 className="font-semibold text-slate-200">
                  Top-5 Predictions
                </h4>
                <div className="space-y-2">
                  {inferenceResults.map((r, idx) => (
                    <div
                      key={`${r.class_idx}-${idx}`}
                      className="flex items-center gap-3 bg-slate-800 rounded-lg p-3 border border-slate-700"
                    >
                      <span className="flex items-center justify-center w-8 h-8 rounded-full bg-indigo-900 text-indigo-300 font-bold text-sm">
                        {idx + 1}
                      </span>
                      <div className="flex-1">
                        <p className="font-medium text-slate-200">
                          {r.class_name}
                        </p>
                        <p className="text-xs text-slate-500">
                          Index {r.class_idx}
                        </p>
                      </div>
                      <span className="font-bold text-indigo-400">
                        {(r.prob * 100).toFixed(2)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
