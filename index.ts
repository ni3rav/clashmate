import { JSDOM } from "jsdom";
import { readFileSync } from "fs";

// Read the HTML file
const html = readFileSync("scrap.html", "utf-8");
const dom = new JSDOM(html);
const document = dom.window.document;

interface Card {
  name: string;
  type: string;
  elixir?: string;
  hitpoints?: string;
  damage?: string;
  hitSpeed?: string;
  dps?: string;
  areaDamage?: string;
  range?: string;
  count?: string;
  deployTime?: string;
  lifetime?: string;
  radius?: string;
  spawnedUnit?: string;
  spawnCondition?: string;
}

const cards: Card[] = [];

// Parse Troops table
console.log("Parsing Troops...");
const troopsTable = document.querySelector("#tpt-1");
if (troopsTable) {
  const rows = troopsTable.querySelectorAll("tbody tr");
  rows.forEach((row) => {
    const cells = row.querySelectorAll("td");
    if (cells.length >= 9) {
      const nameCell = cells[1].querySelector("a");
      if (nameCell) {
        cards.push({
          name: nameCell.textContent?.replace(/\s+/g, " ").trim() || "",
          type: "Troop",
          elixir: cells[2].textContent?.trim(),
          hitpoints: cells[3].textContent?.trim(),
          damage: cells[4].textContent?.trim(),
          hitSpeed: cells[5].textContent?.trim(),
          dps: cells[6].textContent?.trim(),
          areaDamage: cells[7].textContent?.trim(),
          range: cells[8].textContent?.trim(),
          count: cells[9].textContent?.trim(),
        });
      }
    }
  });
}

// Parse Defensive Buildings
console.log("Parsing Defensive Buildings...");
const defensiveTable = document.querySelector("#tpt-2");
if (defensiveTable) {
  const rows = defensiveTable.querySelectorAll("tbody tr");
  rows.forEach((row) => {
    const cells = row.querySelectorAll("td");
    if (cells.length >= 8) {
      const nameCell = cells[1].querySelector("a");
      if (nameCell) {
        cards.push({
          name: nameCell.textContent?.replace(/\s+/g, " ").trim() || "",
          type: "Defensive Building",
          elixir: cells[2].textContent?.trim(),
          hitpoints: cells[3].textContent?.trim(),
          deployTime: cells[4].textContent?.trim(),
          damage: cells[5].textContent?.trim(),
          hitSpeed: cells[6].textContent?.trim(),
          dps: cells[7].textContent?.trim(),
          areaDamage: cells[8].textContent?.trim(),
          range: cells[9].textContent?.trim(),
        });
      }
    }
  });
}

// Parse Passive Buildings
console.log("Parsing Passive Buildings...");
const passiveTable = document.querySelector("#tpt-3");
if (passiveTable) {
  const rows = passiveTable.querySelectorAll("tbody tr");
  rows.forEach((row) => {
    const cells = row.querySelectorAll("td");
    if (cells.length >= 4) {
      const nameCell = cells[1].querySelector("a");
      if (nameCell) {
        cards.push({
          name: nameCell.textContent?.replace(/\s+/g, " ").trim() || "",
          type: "Passive Building",
          elixir: cells[2].textContent?.trim(),
          hitpoints: cells[3].textContent?.trim(),
          areaDamage: cells[4].textContent?.trim(),
          lifetime: cells[5].textContent?.trim(),
        });
      }
    }
  });
}

// Parse Damaging Spells
console.log("Parsing Damaging Spells...");
const spellsTable = document.querySelector("#tpt-4");
if (spellsTable) {
  const rows = spellsTable.querySelectorAll("tbody tr");
  rows.forEach((row) => {
    const cells = row.querySelectorAll("td");
    if (cells.length >= 4) {
      const nameCell = cells[1].querySelector("a");
      if (nameCell) {
        cards.push({
          name: nameCell.textContent?.replace(/\s+/g, " ").trim() || "",
          type: "Spell",
          elixir: cells[2].textContent?.trim(),
          damage: cells[3].textContent?.trim(),
          areaDamage: cells[4].textContent?.trim(),
          radius: cells[5].textContent?.trim(),
        });
      }
    }
  });
}

// Parse Spawners
console.log("Parsing Spawners...");
const spawnersTable = document.querySelector("#tpt-5");
if (spawnersTable) {
  const rows = spawnersTable.querySelectorAll("tbody tr");
  rows.forEach((row) => {
    const cells = row.querySelectorAll("td");
    if (cells.length >= 6) {
      const nameCell = cells[1].querySelector("a");
      if (nameCell) {
        const spawnerType = cells[2].textContent?.replace(/\s+/g, " ").trim();
        const spawnedUnit = cells[4]
          .querySelector("a")
          ?.textContent?.replace(/\s+/g, " ")
          .trim();
        const spawnCondition = cells[5].textContent
          ?.replace(/\s+/g, " ")
          .trim();
        const spawnCount = cells[6].textContent?.trim();

        cards.push({
          name: nameCell.textContent?.replace(/\s+/g, " ").trim() || "",
          type: `Spawner (${spawnerType})`,
          elixir: cells[3].textContent?.trim(),
          spawnedUnit: spawnedUnit,
          spawnCondition: spawnCondition,
          count: spawnCount,
        });
      }
    }
  });
}

console.log(`\nTotal cards parsed: ${cards.length}`);

// Convert to CSV
function escapeCSV(value: string | undefined): string {
  if (!value) return "";
  // Escape quotes and wrap in quotes if contains comma, quote, or newline
  if (value.includes(",") || value.includes('"') || value.includes("\n")) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

// Get all unique fields across all cards
const allFields = new Set<string>();
cards.forEach((card) => {
  Object.keys(card).forEach((key) => allFields.add(key));
});

// Create CSV header
const fields = Array.from(allFields);
const csvHeader = fields.join(",");

// Create CSV rows
const csvRows = cards.map((card) => {
  return fields.map((field) => escapeCSV(card[field as keyof Card])).join(",");
});

const csvContent = [csvHeader, ...csvRows].join("\n");

// Save to CSV file using Bun's API
await Bun.write("./model/cards.csv", csvContent);
console.log("✅ Data saved to model/cards.csv");

// Print summary
const typeCounts: Record<string, number> = {};
cards.forEach((card) => {
  typeCounts[card.type] = (typeCounts[card.type] || 0) + 1;
});

console.log("\n📊 Summary:");
Object.entries(typeCounts).forEach(([type, count]) => {
  console.log(`  ${type}: ${count}`);
});
