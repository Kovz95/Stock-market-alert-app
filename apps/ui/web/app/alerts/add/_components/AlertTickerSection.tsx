"use client";

import { useAtomValue, useSetAtom } from "jotai";
import {
  Field,
  FieldGroup,
  FieldLabel,
  FieldContent,
  FieldLegend,
} from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  addAlertIsRatioAtom,
  addAlertTickerAtom,
  addAlertStockNameAtom,
  addAlertTicker1Atom,
  addAlertTicker2Atom,
  addAlertStockName1Atom,
  addAlertStockName2Atom,
  addAlertAdjustmentMethodAtom,
} from "@/lib/store/add-alert";

const ADJUSTMENT_METHODS = ["None", "split", "dividend"] as const;

export function AlertTickerSection() {
  const isRatio = useAtomValue(addAlertIsRatioAtom);
  const setIsRatio = useSetAtom(addAlertIsRatioAtom);
  const ticker = useAtomValue(addAlertTickerAtom);
  const setTicker = useSetAtom(addAlertTickerAtom);
  const stockName = useAtomValue(addAlertStockNameAtom);
  const setStockName = useSetAtom(addAlertStockNameAtom);
  const ticker1 = useAtomValue(addAlertTicker1Atom);
  const setTicker1 = useSetAtom(addAlertTicker1Atom);
  const ticker2 = useAtomValue(addAlertTicker2Atom);
  const setTicker2 = useSetAtom(addAlertTicker2Atom);
  const stockName1 = useAtomValue(addAlertStockName1Atom);
  const setStockName1 = useSetAtom(addAlertStockName1Atom);
  const stockName2 = useAtomValue(addAlertStockName2Atom);
  const setStockName2 = useSetAtom(addAlertStockName2Atom);
  const adjustmentMethod = useAtomValue(addAlertAdjustmentMethodAtom);
  const setAdjustmentMethod = useSetAtom(addAlertAdjustmentMethodAtom);

  return (
    <FieldGroup>
      <Field>
        <FieldLegend>Ratio of 2 assets?</FieldLegend>
        <FieldContent>
          <Select
            value={isRatio ? "Yes" : "No"}
            onValueChange={(v) => setIsRatio(v === "Yes")}
          >
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="No">No</SelectItem>
              <SelectItem value="Yes">Yes</SelectItem>
            </SelectContent>
          </Select>
        </FieldContent>
      </Field>

      {!isRatio ? (
        <>
          <Field>
            <FieldLabel>Stock / asset name</FieldLabel>
            <FieldContent>
              <Input
                placeholder="e.g. Apple Inc."
                value={stockName}
                onChange={(e) => setStockName(e.target.value)}
              />
            </FieldContent>
          </Field>
          <Field>
            <FieldLabel>Ticker symbol *</FieldLabel>
            <FieldContent>
              <Input
                placeholder="e.g. AAPL"
                value={ticker}
                onChange={(e) => setTicker(e.target.value)}
              />
            </FieldContent>
          </Field>
        </>
      ) : (
        <>
          <Field>
            <FieldLabel>First asset name</FieldLabel>
            <FieldContent>
              <Input
                placeholder="e.g. SPY"
                value={stockName1}
                onChange={(e) => setStockName1(e.target.value)}
              />
            </FieldContent>
          </Field>
          <Field>
            <FieldLabel>First ticker *</FieldLabel>
            <FieldContent>
              <Input
                placeholder="e.g. SPY"
                value={ticker1}
                onChange={(e) => setTicker1(e.target.value)}
              />
            </FieldContent>
          </Field>
          <Field>
            <FieldLabel>Second asset name</FieldLabel>
            <FieldContent>
              <Input
                placeholder="e.g. QQQ"
                value={stockName2}
                onChange={(e) => setStockName2(e.target.value)}
              />
            </FieldContent>
          </Field>
          <Field>
            <FieldLabel>Second ticker *</FieldLabel>
            <FieldContent>
              <Input
                placeholder="e.g. QQQ"
                value={ticker2}
                onChange={(e) => setTicker2(e.target.value)}
              />
            </FieldContent>
          </Field>
          <Field>
            <FieldLegend>Adjustment method (futures/ratio)</FieldLegend>
            <FieldContent>
              <Select
                value={adjustmentMethod || "None"}
                onValueChange={setAdjustmentMethod}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ADJUSTMENT_METHODS.map((m) => (
                    <SelectItem key={m} value={m}>
                      {m}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </FieldContent>
          </Field>
        </>
      )}
    </FieldGroup>
  );
}
