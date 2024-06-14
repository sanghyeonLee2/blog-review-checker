import { END_POINTS } from "../constants/urls";
import postRequest from "./request";

export const fetchFromOuter = (blogUrl) =>
  postRequest(END_POINTS.FETCH_FROM_OUTER, { blogUrl });

export const fetchFromInner = (data) =>
  postRequest(END_POINTS.FETCH_FROM_INNER, data);
