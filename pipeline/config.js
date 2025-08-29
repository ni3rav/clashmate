// clashApi.js
import axios from "axios";
import dotenv from "dotenv";

dotenv.config();
const API_BASE = "https://api.clashroyale.com/v1/";
const TOKEN = process.env.CLASH_API_TOKEN;

const api = axios.create({
  baseURL: API_BASE,
  headers: { Authorization: `Bearer ${TOKEN}` },
});

// Get player profile (cards, trophies, etc.)
export async function getPlayer(playerTag) {
  const tag = encodeURIComponent(playerTag); // e.g. "#ABC123"
  const res = await api.get(`/players/${tag}`);
  return res.data;
}

// Get recent battles (decks + outcomes)
export async function getBattleLog(playerTag) {
  const tag = encodeURIComponent(playerTag);
  const res = await api.get(`/players/${tag}/battlelog`);
  return res.data;
}
