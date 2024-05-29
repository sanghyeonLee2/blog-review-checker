const button = document.getElementById("fetchContent");
button.addEventListener("click", onFetchClick);

function onFetchClick() {
  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    const tabId = tabs[0].id;

    chrome.scripting.executeScript(
      {
        target: { tabId },
        files: ["./scripts/content.js"],
      },
      () => handleScriptInjected(tabId)
    );
  });
}

function handleScriptInjected(tabId) {
  chrome.tabs.sendMessage(
    tabId,
    { action: "fetchBlogContent" },
    handleMessageResponse
  );
}

function handleMessageResponse(response) {
  console.log("Received response:", response);

  const content = response?.content;
  const resultElement = document.getElementById("result");
  const rateResultElement = document.getElementById("rateResult");

  if (content && Array.isArray(content)) {
    const isPromotional = content[0][0];
    const probability = content[1][0][1];

    if (isPromotional === 0) {
      resultElement.textContent = "홍보성이 아닙니다";
      rateResultElement.textContent = `리뷰가 홍보성일 확률 : ${probability.toFixed(
        1
      )}%`;
    } else {
      resultElement.textContent = "홍보성입니다";
      rateResultElement.textContent = `리뷰가 홍보성일 확률 : ${(
        100 - probability
      ).toFixed(1)}%`;
    }
  } else {
    console.error("Invalid response structure:", response);
    resultElement.textContent = "결과를 불러오지 못했습니다.";
    rateResultElement.textContent = "";
  }
}
