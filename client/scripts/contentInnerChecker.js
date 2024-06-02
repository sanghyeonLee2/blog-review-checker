export const handleInnerContentCheck = (request, sendResponse) => {
  const iframe = document.querySelector("iframe");
  if (!iframe) return sendResponse("iframe not found");

  try {
    const doc = iframe.contentDocument || iframe.contentWindow.document;
    const title = doc.querySelector("title")?.innerText || "Title not found";

    const textBlocks = doc.querySelectorAll('span[class^="se-fs-"]');
    const contentText = [...textBlocks].map((el) => el.innerText).join("\n");

    const imgs = doc.querySelectorAll('img[class$="egjs-visible"]');
    const imageUrl = imgs.length > 0 ? imgs[imgs.length - 1].src : null;

    fetch("https://192.168.50.230:3000/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ contentText, imageUrl }),
    })
      .then((res) => res.json())
      .then(sendResponse)
      .catch((err) => {
        console.error(err);
        sendResponse("Error sending data to server");
      });
  } catch (err) {
    console.error(err);
    sendResponse("Unable to access iframe content");
  }
};
