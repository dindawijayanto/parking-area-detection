const byId = (id) => document.getElementById(id);
const recording = byId("recording");
const frame = byId("frame");
const model = byId("model");
const stage = byId("frame-stage");
const image = document.createElement("img");
image.id = "frame-image";
const viewButtons = [...document.querySelectorAll("[data-view]")];
const recordings = new Map();
const dateFormat = new Intl.DateTimeFormat("id-ID", {
  day: "numeric", month: "short", year: "numeric", timeZone: "UTC",
});
const percentFormat = new Intl.NumberFormat("id-ID", { maximumFractionDigits: 1 });
const views = {
  parking: { busy: "Menghitung peta parkir…", alt: "Peta ruang kosong dan area terisi motor" },
  motors: { busy: "Menjalankan YOLO…", alt: "Foto dengan masker motor hasil YOLO" },
  original: { alt: "Foto asli area parkir motor" },
};

let view = "parking";
let currentFrame;
let parkingRequest;
let motorRequest;

async function getJson(path, signal) {
  const response = await fetch(path, { signal });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Permintaan gagal.");
  return data;
}

function playerUrl(endpoint, modelId) {
  const params = new URLSearchParams({ frame: currentFrame.id });
  if (modelId) params.set("model", modelId);
  return `/api/${endpoint}?${params}`;
}

function render() {
  const parking = parkingRequest?.result;
  byId("free-spaces").textContent = parking?.freeSpaces ?? "–";
  byId("occupied-percent").textContent = parking ? `${percentFormat.format(parking.occupiedPercent)}%` : "–";

  const request = view === "parking" ? parkingRequest : view === "motors" ? motorRequest : null;
  const source = request?.result?.image || playerUrl("frame");
  if (image.getAttribute("src") !== source) image.src = source;
  image.alt = request?.result ? views[view].alt : views.original.alt;

  viewButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.view === view)));
  for (const key of Object.keys(views)) byId(`${key === "motors" ? "motor" : key}-legend`).hidden = view !== key;
  byId("motor-status").textContent = motorRequest?.result?.detected === false ? "YOLO tidak menemukan motor pada frame ini." : "";
  const message = request?.error || (request?.pending ? views[view].busy : "");
  byId("frame-message").hidden = !message;
  byId("frame-status").textContent = message;
  byId("spinner").hidden = !request?.pending;
  byId("retry").hidden = !request?.error;
}

function loadPrediction(type) {
  const isParking = type === "parking";
  const previous = isParking ? parkingRequest : motorRequest;
  previous?.controller.abort();
  const request = { controller: new AbortController(), pending: true };
  if (isParking) parkingRequest = request;
  else motorRequest = request;
  render();

  const path = isParking ? playerUrl("analysis", model.value) : playerUrl("motors");
  getJson(path, request.controller.signal).then((result) => {
    request.result = result;
  }).catch((error) => {
    if (error.name !== "AbortError") request.error = error.message;
  }).finally(() => {
    request.pending = false;
    if (!request.controller.signal.aborted) render();
  });
}

function selectFrame() {
  const available = recordings.get(recording.value);
  currentFrame = available[frame.selectedIndex];
  byId("previous").disabled = frame.selectedIndex === 0;
  byId("next").disabled = frame.selectedIndex === available.length - 1;
  byId("frame-time").textContent = `${currentFrame.timeOfDay} · ${currentFrame.timestamp.replace("-", ":")}`;
  stage.style.setProperty("--frame-ratio", `${currentFrame.width} / ${currentFrame.height}`);
  motorRequest?.controller.abort();
  motorRequest = null;
  loadPrediction("parking");
  if (view === "motors") loadPrediction("motors");
}

function selectRecording() {
  frame.replaceChildren(...recordings.get(recording.value).map((item) => (
    new Option(item.timestamp.replace("-", ":"), item.id)
  )));
  selectFrame();
}

viewButtons.forEach((button) => button.addEventListener("click", () => {
  view = button.dataset.view;
  if (view === "motors" && !motorRequest) loadPrediction("motors");
  else render();
}));
recording.addEventListener("change", selectRecording);
frame.addEventListener("change", selectFrame);
model.addEventListener("change", () => loadPrediction("parking"));
byId("retry").addEventListener("click", () => loadPrediction(view));
for (const [id, step] of [["previous", -1], ["next", 1]]) {
  byId(id).addEventListener("click", () => {
    frame.selectedIndex += step;
    selectFrame();
  });
}

getJson("/api/frames").then((catalog) => {
  if (!catalog.frames.length) throw new Error("Frame uji tidak ditemukan.");
  for (const item of catalog.frames) {
    if (!recordings.has(item.recording)) {
      recordings.set(item.recording, []);
      const date = dateFormat.format(new Date(`${item.date}T00:00:00Z`));
      recording.add(new Option(`${item.timeOfDay} · ${date}`, item.recording));
    }
    recordings.get(item.recording).push(item);
  }
  catalog.models.forEach((item) => model.add(new Option(item.name, item.id)));
  byId("frame-count").textContent = `${catalog.frames.length} frame uji`;
  selectRecording();
  stage.prepend(image);
  byId("catalog-status").hidden = true;
  byId("workspace").hidden = false;
}).catch((error) => {
  byId("catalog-status").textContent = error.message;
});
