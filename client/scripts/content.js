chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "fetchPageTitle") {
    const fetchData = async () => {
      console.log("Fetching data for URL:", request.url);
      try {
        const response = await fetch(
          "https://192.168.50.230:3000/blog-crawling",
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ blogUrl: request.url }),
          }
        );

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log("Fetched result:", result);
        // 여기에서 결과를 화면에 표시하는 코드를 추가
        displayResultNextToMatchingAnchor(result, request.url);
        sendResponse(result);
      } catch (err) {
        console.log("Error fetching data:", err);
        sendResponse({ error: err.message });
      }
    };

    fetchData();
    return true; // 비동기 응답을 위해 필요
  }
  if (request.action === "fetchBlogContent") {
    const iframe = document.querySelector("iframe");
    if (iframe) {
      try {
        // iframe 내부의 문서에 접근
        const iframeDocument =
          iframe.contentDocument || iframe.contentWindow.document;

        // title 텍스트 가져오기
        const titleText =
          iframeDocument.querySelector("title")?.innerText || "Title not found";
        console.log("Fetched Title text from iframe:", titleText);

        // 특정 클래스명을 가진 span 요소들 선택
        const contentElements = iframeDocument.querySelectorAll(
          'span[class^="se-fs-"]'
        );
        let contentText = [...contentElements]
          .map((element) => element.innerText)
          .join("\n");
        // 특정 클래스명을 가진 img 요소들 선택
        const imagesElements = iframeDocument.querySelectorAll(
          'img[class$="egjs-visible"]'
        );
        let imageUrl = null;
        if (imagesElements.length > 0) {
          const lastImageElement = imagesElements[imagesElements.length - 1];
          imageUrl = lastImageElement.getAttribute("src");
        }
        console.log("Fetched result:", contentText, imageUrl);
        // 서버로 데이터 전송
        fetch("https://192.168.50.230:3000/process", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ contentText, imageUrl }),
        })
          .then((response) => response.json())
          .then((data) => {
            sendResponse(data); // 서버로부터 받은 데이터를 응답으로 보냄
          })
          .catch((error) => {
            console.error("Error:", error);
            sendResponse("Error sending data to server");
          });
      } catch (error) {
        console.error("Error accessing iframe content:", error);
        sendResponse("Unable to access iframe content");
      }
    } else {
      sendResponse("iframe not found");
    }
    return true; // 비동기 응답을 위해 필요
  }
});

function displayResultNextToMatchingAnchor(result, url) {
  const anchors = document.querySelectorAll("a");
  const matchingAnchor = Array.from(anchors).find(
    (anchor) => anchor.href === url
  );

  const isPromotional = result.content[0][0]; // 홍보성 여부
  const probability = result.content[1][0][1].toFixed(2); // 확률

  if (matchingAnchor) {
    const resultElement = document.createElement("span");
    console.log(result);
    resultElement.innerHTML = `${
      isPromotional === 1 ? "홍보성 입니다." : "홍보성이 아닙니다."
    }<br>확률 : ${probability}%`;

    // 여기에 스타일을 설정합니다.
    resultElement.style.position = "absolute";
    resultElement.style.left = "400px";
    resultElement.style.backgroundColor = "white";
    resultElement.style.color = "black";
    resultElement.style.padding = "5px";
    resultElement.style.fontSize = "10px";
    resultElement.style.zIndex = "1000";
    resultElement.style.top = "30px";
    resultElement.style.fontSize = "16px"; // 명시적으로 font-size를 설정합니다.

    const parentElement = matchingAnchor.closest(".desc");
    if (parentElement) {
      parentElement.insertBefore(resultElement, parentElement.firstChild);
    }
  }
}
