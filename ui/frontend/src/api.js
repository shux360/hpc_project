export async function processImage({ file, method, threads, processes }) {
  const formData = new FormData();
  formData.append("image", file);
  formData.append("method", method);
  formData.append("threads", String(threads || 1));
  formData.append("processes", String(processes || 1));

  const response = await fetch("/api/process", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || "Failed to process image");
  }

  return response.json();
}
