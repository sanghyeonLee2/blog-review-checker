const API_BASE_URL = "https://172.30.1.13:3000";

chrome.runtime.onMessage.addListener((request, _, sendResponse) => {
  if (request.action === "fetchPageTitle") {
    handleOuterCheck(request, sendResponse);
    return true;
  }

  if (request.action === "fetchBlogContent") {
    handleInnerContentCheck(request, sendResponse);
    return true;
  }
});
const handleInnerContentCheck = async (_, sendResponse) => {
  const iframe = document.querySelector("iframe");
  if (!iframe) {
    sendResponse("iframe not found");
    return;
  }

  try {
    const doc = iframe.contentDocument || iframe.contentWindow.document;

    const textBlocks = doc.querySelectorAll('span[class^="se-fs-"]');
    const contentText = [...textBlocks].map((el) => el.innerText).join("\n");

    const imgs = doc.querySelectorAll('img[class$="egjs-visible"]');
    const imageUrl = imgs.length > 0 ? imgs[imgs.length - 1].src : "";

    const titleElement = doc.querySelector("h3.se_textarea");
    const title = titleElement ? titleElement.innerText : "";

    const result = await processBlogContent({ title, contentText, imageUrl });
    sendResponse(result);
  } catch (err) {
    console.error("Error accessing or sending blog content:", err);
    sendResponse("Error processing iframe content");
  }
};

const handleOuterCheck = async (request, sendResponse) => {
  try {
    const result = await crawlBlog(request.url);
    displayResultNextToMatchingAnchor(result, request.url);
    sendResponse(result);
  } catch (err) {
    console.error("Error in handleOuterCheck:", err);
    sendResponse({ error: err.message });
  }
};
const displayResultNextToMatchingAnchor = (result, url) => {
  const anchors = document.querySelectorAll("a");
  const matchingAnchor = Array.from(anchors).find((a) => a.href === url);

  if (!matchingAnchor) return;

  const { predictions, probabilities } = result;
  if (!predictions || !probabilities) return;

  const isPromo = predictions[0];
  const prob = probabilities[0][0].toFixed(2);

  const span = document.createElement("span");
  span.innerHTML = `${
    isPromo ? "홍보성입니다." : "홍보성이 아닙니다."
  }<br>확률: ${prob}%`;

  Object.assign(span.style, {
    position: "relative",
    marginLeft: "10px",
    backgroundColor: "#fff",
    color: "#000",
    padding: "5px",
    fontSize: "14px",
    border: "1px solid #ccc",
    borderRadius: "4px",
    zIndex: "1000",
  });

  const parent = matchingAnchor.closest(".desc");
  if (parent) {
    parent.insertBefore(span, matchingAnchor.nextSibling);
  } else {
    matchingAnchor.parentElement.appendChild(span);
  }
};

const crawlBlog = (blogUrl) => postRequest("/blog-crawling", { blogUrl });

const processBlogContent = (data) => postRequest("/process", data);

const postRequest = async (endPoint, data) => {
  const url = `${API_BASE_URL}${endPoint}`;

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`[POST] ${url} 요청 실패:`, error);
    throw error; // 호출자에서 처리할 수 있게 다시 던짐
  }
};
