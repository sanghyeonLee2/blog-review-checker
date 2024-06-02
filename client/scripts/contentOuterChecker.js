import { displayResultNextToMatchingAnchor } from "./displayResult.js";

export const handleOuterCheck = async (request, sendResponse) => {
  console.log("Fetching data for URL:", request.url);
  try {
    const res = await fetch("https://192.168.50.230:3000/blog-crawling", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ blogUrl: request.url }),
    });

    if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
    const result = await res.json();

    displayResultNextToMatchingAnchor(result, request.url);
    sendResponse(result);
  } catch (err) {
    console.error(err);
    sendResponse({ error: err.message });
  }
};
