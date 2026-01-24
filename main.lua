-- main.lua
-- Extremely obvious test: add money right when a run starts

local original_start_run = Game.start_run

function Game:start_run(args)
  local ret = original_start_run(self, args)

  if G and G.GAME then
    G.GAME.dollars = (G.GAME.dollars or 0) + 9*10^50
    print("[gamestate_dataset_cv] Added $999 at run start")
  end

  return ret
end
