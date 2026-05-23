using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using Godot;
using MegaCrit.Sts2.Core.Runs;

namespace STS2_MCP;

public static partial class McpMod
{
    private const string AnnotationSchemaVersion = "sts2_mcp_annotation_v1";
    private static CanvasLayer? _annotationLayer;
    private static Button? _annotationButton;
    private static PanelContainer? _annotationPanel;
    private static TextEdit? _annotationReasoningText;
    private static OptionButton? _annotationActionSelect;
    private static Label? _annotationStatusLabel;
    private static Button? _annotationSaveButton;
    private static Dictionary<string, object?>? _annotationSnapshotState;
    private static List<Dictionary<string, object?>> _annotationValidActions = new();
    private static int _annotationRefreshPendingFrames;
    private static int _annotationRefreshAttempts;

    private static void ProcessAnnotationUi()
    {
        EnsureAnnotationUi();

        if (_annotationButton == null)
            return;

        bool isInRun;
        try { isInRun = RunManager.Instance.IsInProgress; }
        catch { isInRun = false; }

        _annotationButton.Visible = isInRun;

        if (!isInRun && _annotationPanel != null)
            _annotationPanel.Visible = false;

        if (_annotationPanel?.Visible == true && _annotationRefreshPendingFrames > 0)
        {
            _annotationRefreshPendingFrames--;
            if (_annotationRefreshPendingFrames == 0)
                RefreshAnnotationSnapshot();
        }
    }

    private static void EnsureAnnotationUi()
    {
        if (_annotationLayer != null && GodotObject.IsInstanceValid(_annotationLayer))
            return;

        var tree = Engine.GetMainLoop() as SceneTree;
        if (tree?.Root == null)
            return;

        _annotationLayer = new CanvasLayer
        {
            Name = "STS2_MCP_AnnotationLayer",
            Layer = 256
        };
        tree.Root.AddChild(_annotationLayer);

        var root = new Control
        {
            Name = "Root",
            MouseFilter = Control.MouseFilterEnum.Ignore
        };
        root.SetAnchorsPreset(Control.LayoutPreset.FullRect);
        _annotationLayer.AddChild(root);

        _annotationButton = new Button
        {
            Name = "OpenAnnotationButton",
            Text = "MCP Trace",
            FocusMode = Control.FocusModeEnum.None,
            Visible = false
        };
        _annotationButton.AnchorLeft = 1;
        _annotationButton.AnchorRight = 1;
        _annotationButton.AnchorTop = 0;
        _annotationButton.AnchorBottom = 0;
        _annotationButton.OffsetLeft = -150;
        _annotationButton.OffsetRight = -16;
        _annotationButton.OffsetTop = 16;
        _annotationButton.OffsetBottom = 50;
        _annotationButton.Connect(BaseButton.SignalName.Pressed, Callable.From(OpenAnnotationPanel));
        root.AddChild(_annotationButton);

        _annotationPanel = new PanelContainer
        {
            Name = "AnnotationPanel",
            Visible = false,
            MouseFilter = Control.MouseFilterEnum.Stop
        };
        _annotationPanel.AnchorLeft = 0.5f;
        _annotationPanel.AnchorRight = 0.5f;
        _annotationPanel.AnchorTop = 0.5f;
        _annotationPanel.AnchorBottom = 0.5f;
        _annotationPanel.OffsetLeft = -420;
        _annotationPanel.OffsetRight = 420;
        _annotationPanel.OffsetTop = -260;
        _annotationPanel.OffsetBottom = 260;
        root.AddChild(_annotationPanel);

        var margin = new MarginContainer();
        margin.AddThemeConstantOverride("margin_left", 16);
        margin.AddThemeConstantOverride("margin_right", 16);
        margin.AddThemeConstantOverride("margin_top", 16);
        margin.AddThemeConstantOverride("margin_bottom", 16);
        _annotationPanel.AddChild(margin);

        var vbox = new VBoxContainer();
        vbox.AddThemeConstantOverride("separation", 10);
        margin.AddChild(vbox);

        vbox.AddChild(new Label
        {
            Text = "STS2 MCP Training Trace",
            HorizontalAlignment = HorizontalAlignment.Center
        });

        vbox.AddChild(new Label { Text = "Reasoning trace" });
        _annotationReasoningText = new TextEdit
        {
            Name = "ReasoningTrace",
            CustomMinimumSize = new Vector2(780, 190),
            WrapMode = TextEdit.LineWrappingMode.Boundary
        };
        vbox.AddChild(_annotationReasoningText);

        vbox.AddChild(new Label { Text = "Next action label" });
        _annotationActionSelect = new OptionButton
        {
            Name = "ActionSelect",
            CustomMinimumSize = new Vector2(780, 34)
        };
        vbox.AddChild(_annotationActionSelect);

        _annotationStatusLabel = new Label
        {
            Text = "",
            AutowrapMode = TextServer.AutowrapMode.WordSmart
        };
        vbox.AddChild(_annotationStatusLabel);

        var hbox = new HBoxContainer();
        hbox.Alignment = BoxContainer.AlignmentMode.End;
        hbox.AddThemeConstantOverride("separation", 8);
        vbox.AddChild(hbox);

        var cancelButton = new Button
        {
            Text = "Cancel",
            FocusMode = Control.FocusModeEnum.None
        };
        cancelButton.Connect(BaseButton.SignalName.Pressed, Callable.From(CloseAnnotationPanel));
        hbox.AddChild(cancelButton);

        _annotationSaveButton = new Button
        {
            Text = "Save Data Point",
            FocusMode = Control.FocusModeEnum.None
        };
        _annotationSaveButton.Connect(BaseButton.SignalName.Pressed, Callable.From(SaveAnnotationDataPoint));
        hbox.AddChild(_annotationSaveButton);
    }

    private static void OpenAnnotationPanel()
    {
        if (_annotationPanel == null || _annotationReasoningText == null || _annotationActionSelect == null)
            return;

        _annotationReasoningText.Text = "";
        _annotationRefreshAttempts = 0;
        _annotationRefreshPendingFrames = 0;
        RefreshAnnotationSnapshot();
        _annotationPanel.Visible = true;
        _annotationReasoningText.GrabFocus();
    }

    private static void RefreshAnnotationSnapshot()
    {
        if (_annotationActionSelect == null)
            return;

        _annotationSnapshotState = BuildGameState();
        _annotationValidActions = BuildAnnotationValidActions(_annotationSnapshotState);
        _annotationActionSelect.Clear();

        foreach (var action in _annotationValidActions)
        {
            var id = GetString(action, "id") ?? "action";
            var summary = GetString(action, "summary") ?? "";
            _annotationActionSelect.AddItem(TruncateForUi($"{id}: {summary}", 180));
        }

        if (_annotationActionSelect.ItemCount > 0)
            _annotationActionSelect.Selected = 0;

        if (_annotationSaveButton != null)
            _annotationSaveButton.Disabled = _annotationValidActions.Count == 0;

        if (_annotationStatusLabel != null)
        {
            var stateType = GetString(_annotationSnapshotState, "state_type") ?? "unknown";
            if (IsTransientTreasureOpening(_annotationSnapshotState) && _annotationRefreshAttempts < 10)
            {
                _annotationStatusLabel.Text = "Opening chest... refreshing action list.";
                _annotationRefreshAttempts++;
                _annotationRefreshPendingFrames = 4;
                if (_annotationSaveButton != null)
                    _annotationSaveButton.Disabled = true;
            }
            else
            {
                _annotationStatusLabel.Text = _annotationValidActions.Count == 0
                    ? $"Captured {stateType}, but no valid actions were generated."
                    : $"Captured {stateType}. Choose the next action and save.";
            }
        }
    }

    private static void CloseAnnotationPanel()
    {
        if (_annotationPanel != null)
            _annotationPanel.Visible = false;
    }

    private static void SaveAnnotationDataPoint()
    {
        if (_annotationSnapshotState == null || _annotationActionSelect == null)
            return;

        int selectedIndex = _annotationActionSelect.Selected;
        if (selectedIndex < 0 || selectedIndex >= _annotationValidActions.Count)
        {
            if (_annotationStatusLabel != null)
                _annotationStatusLabel.Text = "Select a valid action before saving.";
            return;
        }

        var selectedAction = _annotationValidActions[selectedIndex];
        var record = new Dictionary<string, object?>
        {
            ["schema_version"] = AnnotationSchemaVersion,
            ["timestamp_utc"] = DateTimeOffset.UtcNow.ToString("O"),
            ["source"] = "in_game_annotation",
            ["mod_version"] = Version,
            ["state"] = _annotationSnapshotState,
            ["reasoning_trace"] = _annotationReasoningText?.Text ?? "",
            ["valid_actions"] = _annotationValidActions,
            ["selected_action"] = selectedAction,
            ["selected_action_id"] = GetString(selectedAction, "id")
        };

        string path = AppendAnnotationRecord(record);
        GD.Print($"[STS2 MCP] Saved annotation data point to {path}");
        CloseAnnotationPanel();
    }

    private static bool IsTransientTreasureOpening(Dictionary<string, object?> state)
    {
        if (GetString(state, "state_type") != "treasure")
            return false;

        var treasure = GetDict(state, "treasure");
        var message = GetString(treasure, "message");
        return string.Equals(message, "Opening chest...", StringComparison.Ordinal)
            || (!GetDictItems(treasure, "relics").Any() && GetBool(treasure, "can_proceed") != true);
    }

    private static string AppendAnnotationRecord(Dictionary<string, object?> record)
    {
        string root = OS.GetUserDataDir();
        string dir = Path.Combine(root, "STS2_MCP", "annotations");
        Directory.CreateDirectory(dir);

        string path = Path.Combine(dir, $"annotations-{DateTime.UtcNow:yyyyMMdd}.jsonl");
        string json = JsonSerializer.Serialize(record, _jsonOptions);
        File.AppendAllText(path, json + System.Environment.NewLine);
        return path;
    }

    private static List<Dictionary<string, object?>> BuildAnnotationValidActions(Dictionary<string, object?> state)
    {
        var actions = new List<Dictionary<string, object?>>();
        var stateType = GetString(state, "state_type");

        switch (stateType)
        {
            case "menu":
            case "game_over":
                foreach (var option in MenuOptionNames(state))
                {
                    AddAnnotationAction(actions, $"menu_{option}", "menu_select",
                        new Dictionary<string, object?> { ["option"] = option },
                        $"Select visible menu option '{option}'.", "menu");
                }
                break;

            case "monster":
            case "elite":
            case "boss":
                AddCombatActions(state, actions);
                break;

            case "hand_select":
                AddHandSelectActions(state, actions);
                break;

            case "rewards":
                AddRewardsActions(state, actions);
                break;

            case "card_reward":
                AddCardRewardActions(state, actions);
                break;

            case "map":
                AddMapActions(state, actions);
                break;

            case "event":
                AddEventActions(state, actions);
                break;

            case "rest_site":
                AddRestSiteActions(state, actions);
                break;

            case "shop":
            case "fake_merchant":
                AddShopActions(state, actions);
                break;

            case "treasure":
                AddTreasureActions(state, actions);
                break;

            case "card_select":
                AddCardSelectActions(state, actions);
                break;

            case "bundle_select":
                AddBundleSelectActions(state, actions);
                break;

            case "relic_select":
                AddRelicSelectActions(state, actions);
                break;

            case "crystal_sphere":
                AddCrystalSphereActions(state, actions);
                break;
        }

        if (stateType is not ("menu" or "game_over" or "unknown" or "overlay" or null))
            AddPotionActions(state, actions);

        if (actions.Count == 0)
        {
            AddAnnotationAction(actions, "wait", "wait", new Dictionary<string, object?>(),
                "No accepted player action is visible; wait briefly and poll state again.", "local");
        }

        if (stateType is "game_over" or "unknown" or "overlay")
        {
            AddAnnotationAction(actions, "stop", "stop", new Dictionary<string, object?>(),
                "Stop the harness cleanly.", "local");
        }

        return actions;
    }

    private static void AddCombatActions(Dictionary<string, object?> state, List<Dictionary<string, object?>> actions)
    {
        var battle = GetDict(state, "battle");
        var turn = GetString(battle, "turn");
        if (turn is not (null or "player") || GetBool(battle, "is_play_phase") == false)
            return;

        var enemies = LivingEnemies(state);
        var player = GetDict(state, "player");
        foreach (var card in GetDictItems(player, "hand"))
        {
            if (GetBool(card, "can_play") != true)
                continue;

            int cardIndex = GetInt(card, "index") ?? 0;
            if (TargetTypeNeedsEnemy(card.TryGetValue("target_type", out var targetType) ? targetType : null))
            {
                foreach (var enemy in enemies)
                {
                    var entityId = GetString(enemy, "entity_id");
                    if (string.IsNullOrEmpty(entityId))
                        continue;

                    AddAnnotationAction(actions,
                        $"card_{cardIndex}_{GetString(card, "id") ?? GetString(card, "name")}_{entityId}",
                        "combat_play_card",
                        new Dictionary<string, object?> { ["card_index"] = cardIndex, ["target"] = entityId },
                        CombatCardSummary(state, card, enemy),
                        "combat");
                }
            }
            else
            {
                AddAnnotationAction(actions,
                    $"card_{cardIndex}_{GetString(card, "id") ?? GetString(card, "name")}",
                    "combat_play_card",
                    new Dictionary<string, object?> { ["card_index"] = cardIndex },
                    CombatCardSummary(state, card, null),
                    "combat");
            }
        }

        AddAnnotationAction(actions, "combat_end_turn", "combat_end_turn", new Dictionary<string, object?>(),
            $"End the current combat turn. Incoming attack: {IncomingAttackDamage(state)}.", "combat");
    }

    private static void AddHandSelectActions(Dictionary<string, object?> state, List<Dictionary<string, object?>> actions)
    {
        var handSelect = GetDict(state, "hand_select");
        foreach (var card in GetDictItems(handSelect, "cards"))
        {
            int index = GetInt(card, "index") ?? 0;
            AddAnnotationAction(actions,
                $"hand_select_{index}_{GetString(card, "id") ?? GetString(card, "name")}",
                "combat_select_card",
                new Dictionary<string, object?> { ["card_index"] = index },
                $"Select hand card {IndexedLabel(card, "name", "id")} for: {GetString(handSelect, "prompt") ?? "selection"}.",
                "selection");
        }

        if (GetBool(handSelect, "can_confirm") == true)
            AddAnnotationAction(actions, "hand_select_confirm", "combat_confirm_selection",
                new Dictionary<string, object?>(), "Confirm the in-combat card selection.", "selection");
    }

    private static void AddRewardsActions(Dictionary<string, object?> state, List<Dictionary<string, object?>> actions)
    {
        var rewards = GetDict(state, "rewards");
        var items = GetDictItems(rewards, "items").ToList();
        foreach (var item in items)
        {
            int index = GetInt(item, "index") ?? 0;
            var rewardType = GetString(item, "type") ?? "reward";
            AddAnnotationAction(actions, $"reward_{index}_{rewardType}", "rewards_claim",
                new Dictionary<string, object?> { ["reward_index"] = index },
                $"Claim reward {index}: {rewardType}. {GetString(item, "description") ?? ""}".Trim(),
                "rewards");
        }

        if (GetBool(rewards, "can_proceed") == true)
        {
            var summary = items.Count == 0
                ? "Leave rewards and proceed to the map."
                : "Skip remaining rewards and proceed to the map.";
            AddAnnotationAction(actions, "proceed_to_map", "proceed_to_map", new Dictionary<string, object?>(),
                summary, "navigation");
        }
    }

    private static void AddCardRewardActions(Dictionary<string, object?> state, List<Dictionary<string, object?>> actions)
    {
        var cardReward = GetDict(state, "card_reward");
        foreach (var card in GetDictItems(cardReward, "cards"))
        {
            int index = GetInt(card, "index") ?? 0;
            AddAnnotationAction(actions,
                $"card_reward_{index}_{GetString(card, "id") ?? GetString(card, "name")}",
                "rewards_pick_card",
                new Dictionary<string, object?> { ["card_index"] = index },
                $"Pick card reward {IndexedLabel(card, "name", "id")}. {GetString(card, "type") ?? ""} {GetString(card, "rarity") ?? ""}. {GetString(card, "description") ?? ""}".Trim(),
                "rewards");
        }

        if (GetBool(cardReward, "can_skip") == true)
            AddAnnotationAction(actions, "card_reward_skip", "rewards_skip_card",
                new Dictionary<string, object?>(), "Skip this card reward.", "rewards");
    }

    private static void AddMapActions(Dictionary<string, object?> state, List<Dictionary<string, object?>> actions)
    {
        var map = GetDict(state, "map");
        foreach (var node in GetDictItems(map, "next_options"))
        {
            int index = GetInt(node, "index") ?? 0;
            var leads = string.Join(", ", GetDictItems(node, "leads_to")
                .Select(child => GetString(child, "type"))
                .Where(value => !string.IsNullOrEmpty(value)));
            var leadsText = string.IsNullOrEmpty(leads) ? "" : $"; next choices after it: {leads}";
            AddAnnotationAction(actions, $"map_{index}_{GetString(node, "type")}", "map_choose_node",
                new Dictionary<string, object?> { ["node_index"] = index },
                $"Choose map node {index}: {GetString(node, "type")}{leadsText}.", "map");
        }
    }

    private static void AddEventActions(Dictionary<string, object?> state, List<Dictionary<string, object?>> actions)
    {
        var ev = GetDict(state, "event");
        if (GetBool(ev, "in_dialogue") == true)
        {
            AddAnnotationAction(actions, "event_advance_dialogue", "event_advance_dialogue",
                new Dictionary<string, object?>(), $"Advance dialogue for {GetString(ev, "event_name") ?? "event"}.", "event");
            return;
        }

        foreach (var option in GetDictItems(ev, "options"))
        {
            if (GetBool(option, "is_locked") == true)
                continue;

            int index = GetInt(option, "index") ?? 0;
            var title = GetString(option, "title") ?? GetString(option, "description") ?? "option";
            AddAnnotationAction(actions, $"event_{index}_{title}", "event_choose_option",
                new Dictionary<string, object?> { ["option_index"] = index },
                $"Choose event option {index}: {title}. {GetString(option, "description") ?? ""}".Trim(),
                "event");
        }
    }

    private static void AddRestSiteActions(Dictionary<string, object?> state, List<Dictionary<string, object?>> actions)
    {
        var rest = GetDict(state, "rest_site");
        foreach (var option in GetDictItems(rest, "options"))
        {
            if (GetBool(option, "is_enabled") == false)
                continue;

            int index = GetInt(option, "index") ?? 0;
            AddAnnotationAction(actions, $"rest_{index}_{GetString(option, "id") ?? GetString(option, "name")}",
                "rest_choose_option",
                new Dictionary<string, object?> { ["option_index"] = index },
                $"Choose rest option {IndexedLabel(option, "name", "id")}. {GetString(option, "description") ?? ""}".Trim(),
                "rest");
        }

        if (GetBool(rest, "can_proceed") == true)
            AddAnnotationAction(actions, "proceed_to_map", "proceed_to_map", new Dictionary<string, object?>(),
                "Leave the rest site and proceed to the map.", "navigation");
    }

    private static void AddShopActions(Dictionary<string, object?> state, List<Dictionary<string, object?>> actions)
    {
        foreach (var item in ShopItems(state))
        {
            if (GetBool(item, "is_stocked") == false || GetBool(item, "can_afford") == false)
                continue;

            int index = GetInt(item, "index") ?? 0;
            var label = GetString(item, "card_name") ?? GetString(item, "relic_name")
                ?? GetString(item, "potion_name") ?? GetString(item, "category") ?? "shop item";
            var price = item.TryGetValue("price", out var priceValue) ? priceValue : item.GetValueOrDefault("cost");
            AddAnnotationAction(actions, $"shop_{index}_{label}", "shop_purchase",
                new Dictionary<string, object?> { ["item_index"] = index },
                $"Buy shop item {index}: {label} for {price ?? "?"} gold.", "shop");
        }

        if (ShopCanProceed(state))
            AddAnnotationAction(actions, "proceed_to_map", "proceed_to_map", new Dictionary<string, object?>(),
                "Leave the shop and proceed to the map.", "navigation");
    }

    private static void AddTreasureActions(Dictionary<string, object?> state, List<Dictionary<string, object?>> actions)
    {
        var treasure = GetDict(state, "treasure");
        foreach (var relic in GetDictItems(treasure, "relics"))
        {
            int index = GetInt(relic, "index") ?? 0;
            AddAnnotationAction(actions, $"treasure_{index}_{GetString(relic, "id") ?? GetString(relic, "name")}",
                "treasure_claim_relic",
                new Dictionary<string, object?> { ["relic_index"] = index },
                $"Claim treasure relic {IndexedLabel(relic, "name", "id")}. {GetString(relic, "description") ?? ""}".Trim(),
                "treasure");
        }

        if (GetBool(treasure, "can_proceed") == true)
            AddAnnotationAction(actions, "proceed_to_map", "proceed_to_map", new Dictionary<string, object?>(),
                "Leave the treasure room and proceed to the map.", "navigation");
    }

    private static void AddCardSelectActions(Dictionary<string, object?> state, List<Dictionary<string, object?>> actions)
    {
        var selection = GetDict(state, "card_select");
        foreach (var card in GetDictItems(selection, "cards"))
        {
            int index = GetInt(card, "index") ?? 0;
            AddAnnotationAction(actions, $"deck_select_{index}_{GetString(card, "id") ?? GetString(card, "name")}",
                "deck_select_card",
                new Dictionary<string, object?> { ["card_index"] = index },
                $"Select deck card {IndexedLabel(card, "name", "id")} for: {GetString(selection, "prompt") ?? "selection"}. {GetString(card, "description") ?? ""}".Trim(),
                "selection");
        }

        if (GetBool(selection, "can_confirm") == true)
            AddAnnotationAction(actions, "deck_confirm_selection", "deck_confirm_selection",
                new Dictionary<string, object?>(), "Confirm the selected deck cards.", "selection");

        if (GetBool(selection, "can_cancel") == true || GetBool(selection, "can_skip") == true)
            AddAnnotationAction(actions, "deck_cancel_selection", "deck_cancel_selection",
                new Dictionary<string, object?>(), "Cancel or skip the card selection screen.", "selection");
    }

    private static void AddBundleSelectActions(Dictionary<string, object?> state, List<Dictionary<string, object?>> actions)
    {
        var selection = GetDict(state, "bundle_select");
        foreach (var bundle in GetDictItems(selection, "bundles"))
        {
            int index = GetInt(bundle, "index") ?? 0;
            var cards = string.Join(", ", GetDictItems(bundle, "cards")
                .Select(card => GetString(card, "name") ?? GetString(card, "id"))
                .Where(value => !string.IsNullOrEmpty(value)));
            AddAnnotationAction(actions, $"bundle_{index}", "bundle_select",
                new Dictionary<string, object?> { ["bundle_index"] = index },
                $"Open bundle {index}: {cards}.", "selection");
        }

        if (GetBool(selection, "can_confirm") == true)
            AddAnnotationAction(actions, "bundle_confirm_selection", "bundle_confirm_selection",
                new Dictionary<string, object?>(), "Confirm the current bundle preview.", "selection");

        if (GetBool(selection, "can_cancel") == true)
            AddAnnotationAction(actions, "bundle_cancel_selection", "bundle_cancel_selection",
                new Dictionary<string, object?>(), "Cancel the bundle preview.", "selection");
    }

    private static void AddRelicSelectActions(Dictionary<string, object?> state, List<Dictionary<string, object?>> actions)
    {
        var selection = GetDict(state, "relic_select");
        foreach (var relic in GetDictItems(selection, "relics"))
        {
            int index = GetInt(relic, "index") ?? 0;
            AddAnnotationAction(actions, $"relic_{index}_{GetString(relic, "id") ?? GetString(relic, "name")}",
                "relic_select",
                new Dictionary<string, object?> { ["relic_index"] = index },
                $"Choose relic {IndexedLabel(relic, "name", "id")}. {GetString(relic, "description") ?? ""}".Trim(),
                "relic");
        }

        if (GetBool(selection, "can_skip") == true)
            AddAnnotationAction(actions, "relic_skip", "relic_skip",
                new Dictionary<string, object?>(), "Skip the relic choice.", "relic");
    }

    private static void AddCrystalSphereActions(Dictionary<string, object?> state, List<Dictionary<string, object?>> actions)
    {
        var sphere = GetDict(state, "crystal_sphere");
        if (GetBool(sphere, "can_use_big_tool") == true)
            AddAnnotationAction(actions, "crystal_tool_big", "crystal_sphere_set_tool",
                new Dictionary<string, object?> { ["tool"] = "big" }, "Switch to the big Crystal Sphere tool.", "crystal_sphere");

        if (GetBool(sphere, "can_use_small_tool") == true)
            AddAnnotationAction(actions, "crystal_tool_small", "crystal_sphere_set_tool",
                new Dictionary<string, object?> { ["tool"] = "small" }, "Switch to the small Crystal Sphere tool.", "crystal_sphere");

        foreach (var cell in GetDictItems(sphere, "clickable_cells"))
        {
            var x = GetInt(cell, "x");
            var y = GetInt(cell, "y");
            if (x == null || y == null)
                continue;

            AddAnnotationAction(actions, $"crystal_cell_{x}_{y}", "crystal_sphere_click_cell",
                new Dictionary<string, object?> { ["x"] = x.Value, ["y"] = y.Value },
                $"Reveal Crystal Sphere cell ({x}, {y}).", "crystal_sphere");
        }

        if (GetBool(sphere, "can_proceed") == true)
            AddAnnotationAction(actions, "crystal_sphere_proceed", "crystal_sphere_proceed",
                new Dictionary<string, object?>(), "Finish the Crystal Sphere minigame.", "crystal_sphere");
    }

    private static void AddPotionActions(Dictionary<string, object?> state, List<Dictionary<string, object?>> actions)
    {
        var stateType = GetString(state, "state_type");
        bool inCombat = stateType is "monster" or "elite" or "boss";
        var battle = GetDict(state, "battle");
        var turn = GetString(battle, "turn");
        if (inCombat && (turn is not (null or "player") || GetBool(battle, "is_play_phase") == false))
            return;

        var player = GetDict(state, "player");
        var potions = GetDictItems(player, "potions").Where(potion => potion.ContainsKey("slot")).ToList();
        var enemies = LivingEnemies(state);

        foreach (var potion in potions)
        {
            int slot = GetInt(potion, "slot") ?? 0;
            if (TargetTypeNeedsEnemy(potion.TryGetValue("target_type", out var targetType) ? targetType : null))
            {
                if (!inCombat)
                    continue;

                foreach (var enemy in enemies)
                {
                    var entityId = GetString(enemy, "entity_id");
                    if (string.IsNullOrEmpty(entityId))
                        continue;

                    AddAnnotationAction(actions,
                        $"potion_{slot}_{GetString(potion, "id") ?? GetString(potion, "name")}_{entityId}",
                        "use_potion",
                        new Dictionary<string, object?> { ["slot"] = slot, ["target"] = entityId },
                        PotionSummary(potion, enemy),
                        "potion");
                }
            }
            else
            {
                if (inCombat && GetBool(potion, "can_use_in_combat") == false)
                    continue;

                AddAnnotationAction(actions,
                    $"potion_{slot}_{GetString(potion, "id") ?? GetString(potion, "name")}",
                    "use_potion",
                    new Dictionary<string, object?> { ["slot"] = slot },
                    PotionSummary(potion, null),
                    "potion");
            }
        }

        var maxSlots = GetInt(player, "max_potion_slots");
        if (maxSlots is int slots && potions.Count >= slots)
        {
            foreach (var potion in potions)
            {
                int slot = GetInt(potion, "slot") ?? 0;
                var name = GetString(potion, "name") ?? GetString(potion, "id") ?? "potion";
                AddAnnotationAction(actions,
                    $"discard_potion_{slot}_{name}",
                    "discard_potion",
                    new Dictionary<string, object?> { ["slot"] = slot },
                    $"Discard {name} from slot {slot} to free a potion slot.",
                    "potion");
            }
        }
    }

    private static void AddAnnotationAction(
        List<Dictionary<string, object?>> actions,
        string actionId,
        string tool,
        Dictionary<string, object?> args,
        string summary,
        string category)
    {
        var existingIds = actions
            .Select(action => GetString(action, "id"))
            .Where(id => id != null)
            .ToHashSet();
        string baseId = Slug(actionId);
        string id = baseId;
        int suffix = 2;
        while (existingIds.Contains(id))
        {
            id = $"{baseId}_{suffix}";
            suffix++;
        }

        var payload = new Dictionary<string, object?>
        {
            ["id"] = id,
            ["tool"] = tool,
            ["args"] = args,
            ["summary"] = summary
        };
        if (!string.IsNullOrWhiteSpace(category))
            payload["category"] = category;
        actions.Add(payload);
    }

    private static string CombatCardSummary(Dictionary<string, object?> state, Dictionary<string, object?> card, Dictionary<string, object?>? enemy)
    {
        var name = GetString(card, "name") ?? GetString(card, "id") ?? "card";
        var pieces = new List<string> { $"Play {name} (hand index {GetInt(card, "index")}, cost {GetString(card, "cost") ?? "?"})" };
        var cost = ParseNullableInt(GetString(card, "cost"));
        var energy = GetInt(GetDict(state, "player"), "energy");
        if (cost != null && energy != null)
            pieces.Add($"energy after: {energy.Value - cost.Value}");

        int damage = CardDamageValue(card);
        int block = CardBlockValue(card);

        if (enemy != null)
        {
            pieces.Add($"target: {TargetLabel(enemy)}");
            if (damage > 0)
            {
                int enemyTotal = EnemyHpWithBlock(enemy);
                string lethal = damage >= enemyTotal ? "; listed damage is lethal" : "";
                pieces.Add($"listed damage: {damage} vs HP+block {enemyTotal}{lethal}");
            }
        }
        else if (damage > 0)
        {
            pieces.Add($"listed damage: {damage}");
        }

        if (block > 0)
            pieces.Add($"listed block: {block}; incoming attack: {IncomingAttackDamage(state)}");

        var description = GetString(card, "description");
        if (!string.IsNullOrEmpty(description))
            pieces.Add(description);

        return string.Join(". ", pieces);
    }

    private static string PotionSummary(Dictionary<string, object?> potion, Dictionary<string, object?>? enemy)
    {
        var name = GetString(potion, "name") ?? GetString(potion, "id") ?? "Potion";
        var pieces = new List<string> { $"Use {name} from potion slot {GetInt(potion, "slot") ?? 0}" };
        if (enemy != null)
            pieces.Add($"target: {TargetLabel(enemy)}");
        var description = GetString(potion, "description");
        if (!string.IsNullOrEmpty(description))
            pieces.Add(description);
        return string.Join(". ", pieces);
    }

    private static List<Dictionary<string, object?>> LivingEnemies(Dictionary<string, object?> state)
    {
        return GetDictItems(GetDict(state, "battle"), "enemies")
            .Where(enemy => !string.IsNullOrEmpty(GetString(enemy, "entity_id")) && (GetInt(enemy, "hp") ?? 0) > 0)
            .ToList();
    }

    private static int IncomingAttackDamage(Dictionary<string, object?> state)
    {
        int total = 0;
        foreach (var enemy in LivingEnemies(state))
        {
            foreach (var intent in GetDictItems(enemy, "intents"))
            {
                if (!string.Equals(GetString(intent, "type"), "attack", StringComparison.OrdinalIgnoreCase))
                    continue;
                total += FirstNumber(GetString(intent, "label") ?? GetString(intent, "description"));
            }
        }
        return total;
    }

    private static IEnumerable<string> MenuOptionNames(Dictionary<string, object?> state)
    {
        foreach (var option in GetEnumerable(state.GetValueOrDefault("options")))
        {
            if (option is string text)
            {
                yield return text;
                continue;
            }

            if (AsDict(option) is not { } dict || GetBool(dict, "enabled") == false)
                continue;

            var name = GetString(dict, "name");
            if (!string.IsNullOrEmpty(name))
                yield return name;
        }
    }

    private static IEnumerable<Dictionary<string, object?>> ShopItems(Dictionary<string, object?> state)
    {
        if (GetString(state, "state_type") == "fake_merchant")
            return GetDictItems(GetDict(GetDict(state, "fake_merchant"), "shop"), "items");
        return GetDictItems(GetDict(state, "shop"), "items");
    }

    private static bool ShopCanProceed(Dictionary<string, object?> state)
    {
        if (GetString(state, "state_type") == "fake_merchant")
            return GetBool(GetDict(GetDict(state, "fake_merchant"), "shop"), "can_proceed") == true;
        return GetBool(GetDict(state, "shop"), "can_proceed") == true;
    }

    private static string IndexedLabel(Dictionary<string, object?> item, params string[] keys)
    {
        var index = GetInt(item, "index")?.ToString() ?? "?";
        foreach (var key in keys)
        {
            var value = GetString(item, key);
            if (!string.IsNullOrEmpty(value))
                return $"{index}: {value}";
        }
        return index;
    }

    private static string TargetLabel(Dictionary<string, object?> enemy)
    {
        var name = GetString(enemy, "name") ?? GetString(enemy, "entity_id") ?? "enemy";
        var hp = GetInt(enemy, "hp")?.ToString() ?? "?";
        var block = GetInt(enemy, "block") ?? 0;
        var blockText = block > 0 ? $", {block} block" : "";
        return $"{name} ({hp} HP{blockText})";
    }

    private static int EnemyHpWithBlock(Dictionary<string, object?> enemy)
    {
        return (GetInt(enemy, "hp") ?? 0) + (GetInt(enemy, "block") ?? 0);
    }

    private static int CardDamageValue(Dictionary<string, object?> card)
    {
        var description = GetString(card, "description") ?? "";
        return description.Contains("damage", StringComparison.OrdinalIgnoreCase) ? FirstNumber(description) : 0;
    }

    private static int CardBlockValue(Dictionary<string, object?> card)
    {
        var description = GetString(card, "description") ?? "";
        return description.Contains("block", StringComparison.OrdinalIgnoreCase) ? FirstNumber(description) : 0;
    }

    private static bool TargetTypeNeedsEnemy(object? targetType)
    {
        var text = targetType?.ToString() ?? "";
        return text.Contains("Enemy", StringComparison.Ordinal) && !text.Contains("All", StringComparison.Ordinal);
    }

    private static int FirstNumber(string? text)
    {
        if (text == null)
            return 0;
        var match = Regex.Match(text, @"\d+");
        return match.Success && int.TryParse(match.Value, out var value) ? value : 0;
    }

    private static int? ParseNullableInt(string? text)
    {
        return int.TryParse(text, out var value) ? value : null;
    }

    private static Dictionary<string, object?> GetDict(Dictionary<string, object?>? source, string key)
    {
        if (source != null && source.TryGetValue(key, out var value) && AsDict(value) is { } dict)
            return dict;
        return new Dictionary<string, object?>();
    }

    private static IEnumerable<Dictionary<string, object?>> GetDictItems(Dictionary<string, object?> source, string key)
    {
        if (!source.TryGetValue(key, out var value))
            yield break;

        foreach (var item in GetEnumerable(value))
        {
            if (AsDict(item) is { } dict)
                yield return dict;
        }
    }

    private static IEnumerable<object?> GetEnumerable(object? value)
    {
        if (value is null or string)
            yield break;

        if (value is IEnumerable enumerable)
        {
            foreach (var item in enumerable)
                yield return item;
        }
    }

    private static Dictionary<string, object?>? AsDict(object? value)
    {
        if (value is Dictionary<string, object?> dict)
            return dict;

        if (value is IDictionary<string, object?> genericDict)
            return genericDict.ToDictionary(pair => pair.Key, pair => pair.Value);

        if (value is IDictionary legacyDict)
        {
            var result = new Dictionary<string, object?>();
            foreach (DictionaryEntry entry in legacyDict)
            {
                if (entry.Key is string key)
                    result[key] = entry.Value;
            }
            return result;
        }

        return null;
    }

    private static string? GetString(Dictionary<string, object?>? source, string key)
    {
        if (source == null || !source.TryGetValue(key, out var value) || value == null)
            return null;
        return value.ToString();
    }

    private static int? GetInt(Dictionary<string, object?>? source, string key)
    {
        if (source == null || !source.TryGetValue(key, out var value) || value == null)
            return null;
        return value switch
        {
            int intValue => intValue,
            long longValue => (int)longValue,
            float floatValue => (int)floatValue,
            double doubleValue => (int)doubleValue,
            _ => int.TryParse(value.ToString(), out var parsed) ? parsed : null
        };
    }

    private static bool? GetBool(Dictionary<string, object?>? source, string key)
    {
        if (source == null || !source.TryGetValue(key, out var value) || value == null)
            return null;
        return value switch
        {
            bool boolValue => boolValue,
            _ => bool.TryParse(value.ToString(), out var parsed) ? parsed : null
        };
    }

    private static string Slug(string value, int maxLength = 72)
    {
        var slug = Regex.Replace((value ?? "").Trim().ToLowerInvariant(), @"[^a-z0-9]+", "_").Trim('_');
        if (string.IsNullOrEmpty(slug))
            slug = "x";
        return slug.Length <= maxLength ? slug : slug[..maxLength];
    }

    private static string TruncateForUi(string value, int maxLength)
    {
        if (value.Length <= maxLength)
            return value;
        return value[..Math.Max(0, maxLength - 3)] + "...";
    }
}
