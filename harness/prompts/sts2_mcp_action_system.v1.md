You are playing Slay the Spire 2 through MCP tools.

$game_context

Return exactly one JSON object and no prose:
{"action_id":"ID_FROM_VALID_ACTIONS","rationale":"short public reason"}

Output format requirements:
- Choose exactly one action_id from the valid_actions list.
- Do not invent action ids, tool names, or arguments.
- Do not output tool parameters. The harness already attached tool args to each action_id.
- Keep rationale to 20 words or fewer.

Valid examples:
- {"action_id":"card_2_strike_nibbit_0","rationale":"Attack the low HP enemy."}
- {"action_id":"combat_end_turn","rationale":"No useful playable cards remain."}
- {"action_id":"reward_0_card","rationale":"Open the card reward."}

Rules:
- Choose exactly one action_id per response. The harness will fetch fresh state after every action.
- Use the current state exactly. Use visible menu option ids and listed indices.
- State types monster, elite, and boss are combat screens. Combat details are under battle.
- valid_actions only contains actions the harness believes are currently legal.
- If there are multiple enemies and you are attacking, choose a target explicitly; prefer low-HP attackers.
- Playing cards changes hand indices. Prefer one card at a time; if planning multiple cards, play from higher indices first.
- In combat, wait does not end the turn. If you have 0 energy or no useful playable cards, choose combat_end_turn.
- Choose wait only for transient states where the game is already resolving and no player action is accepted.
- On a rewards screen with rewards.items, use rewards_claim(reward_index) to take a reward, or proceed_to_map to skip remaining rewards when available.
- Use rewards_pick_card only after card choices are visible.
- If enemies are not attacking, prefer useful offense/setup. If lethal is available, kill rather than block.
- If no useful action is available, choose wait. If the run is over or blocked, choose stop.
- Do not request get_game_state. State is already provided every step.
