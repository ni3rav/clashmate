// pipeline.js
import { getPlayer, getBattleLog } from "./config.js";

async function main() {
  const playerTag = "#"; // replace with real player tag

  // 1. Player profile
  const player = await getPlayer(playerTag);
  console.log("Player trophies:", player.trophies);
  console.log("First few cards:", player.cards.slice(0, 5));

  // 2. Battle logs
  const battles = await getBattleLog(playerTag);

  const battleData = battles.map((b) => {
    const teamDeck = b.team[0].cards.map((c) => c.name);
    const win = b.team[0].crowns > b.opponent[0].crowns ? 1 : 0;
    return { deck: teamDeck, win };
  });

  console.log("Recent battles:", battleData);
}

main();
