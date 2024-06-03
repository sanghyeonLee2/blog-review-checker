const button = document.getElementById("fetchContent");
const buttonText = document.getElementById("buttonText");
const resultElement = document.getElementById("result");
const rateResultElement = document.getElementById("rateResult");

const onFetchClick = () => {
  updateResult("분석 중입니다...", "잠시만 기다려주세요.");
  buttonText.textContent = "분석 중...";
  button.disabled = true;

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const tabId = tabs[0].id;
    chrome.scripting.executeScript(
      {
        target: { tabId },
        files: ["scripts/content.js"],
      },
      () => {
        chrome.tabs.sendMessage(
          tabId,
          { action: "fetchBlogContent" },
          handleMessageResponse
        );
      }
    );
  });
};

button.addEventListener("click", onFetchClick);

const handleMessageResponse = (response) => {
  console.log("Received response:", response);
  buttonText.textContent = "홍보성 체크";
  button.disabled = false;

  const { predictions, probabilities } = response ?? {};

  if (!Array.isArray(predictions) || !Array.isArray(probabilities)) {
    updateResult("결과를 불러오지 못했습니다.");
    return;
  }

  const isPromotional = predictions[0];
  const probability = probabilities[0][0];

  if (isPromotional === 1) {
    updateResult(
      "홍보성입니다",
      `리뷰가 홍보성일 확률 : ${probability.toFixed(1)}%`
    );
  } else {
    updateResult(
      "홍보성이 아닙니다",
      `리뷰가 홍보성일 확률 : ${(100 - probability).toFixed(1)}%`
    );
  }
};

function updateResult(mainText, subText = "") {
  resultElement.textContent = mainText;
  rateResultElement.textContent = subText;
}
