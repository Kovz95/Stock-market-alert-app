"use client";

import { useAtomValue, useSetAtom } from "jotai";
import {
  Field,
  FieldGroup,
  FieldLabel,
  FieldContent,
  FieldLegend,
} from "@/components/ui/field";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { type ConditionEntry } from "./types";
import { ConditionBuilder } from "./ConditionBuilder";
import { ConditionRow } from "./ConditionRow";
import {
  addAlertConditionsAtom,
  addAlertCombinationLogicAtom,
} from "@/lib/store/add-alert";

export function AlertConditionsSection() {
  const conditions = useAtomValue(addAlertConditionsAtom);
  const setConditions = useSetAtom(addAlertConditionsAtom);
  const combinationLogic = useAtomValue(addAlertCombinationLogicAtom);
  const setCombinationLogic = useSetAtom(addAlertCombinationLogicAtom);

  const handleAdd = (entry: ConditionEntry) => {
    setConditions([...conditions, entry]);
  };

  const handleRemove = (id: string) => {
    setConditions(conditions.filter((c) => c.id !== id));
  };

  return (
    <FieldGroup>
      <Field>
        <FieldLegend>Combine conditions with</FieldLegend>
        <FieldContent>
          <Select
            value={combinationLogic}
            onValueChange={(v) => setCombinationLogic(v as "AND" | "OR")}
          >
            <SelectTrigger className="w-full max-w-[120px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="AND">AND</SelectItem>
              <SelectItem value="OR">OR</SelectItem>
            </SelectContent>
          </Select>
        </FieldContent>
      </Field>

      {conditions.length > 0 && (
        <Field>
          <FieldLabel>Current conditions</FieldLabel>
          <FieldContent className="space-y-2">
            {conditions.map((entry) => (
              <ConditionRow
                key={entry.id}
                entry={entry}
                onRemove={() => handleRemove(entry.id)}
              />
            ))}
          </FieldContent>
        </Field>
      )}

      <ConditionBuilder onAdd={handleAdd} />
    </FieldGroup>
  );
}
