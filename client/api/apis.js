import { postRequest } from "./postRequest.js";

export const crawlBlog = (blogUrl) =>
  postRequest("/blog-crawling", { blogUrl });

export const processBlogContent = (data) => postRequest("/process", data);
