<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0-Prizren" styleCategories="Symbology|Labeling">
  <renderer-v2 type="RuleRenderer" symbollevels="0" enableorderby="0" forceraster="0" referencescale="-1">
    <rules key="{root}">
      <rule key="{r0}" label="380 kV station, footprint (OpenStreetMap)" symbol="0" filter="(&quot;max_kv&quot; &gt;= 300) AND &quot;source&quot; &lt;&gt; 'box'" scalemaxdenom="70000" scalemindenom="1" />
      <rule key="{r1}" label="380 kV station, box (no footprint)" symbol="1" filter="(&quot;max_kv&quot; &gt;= 300) AND &quot;source&quot; = 'box'" scalemaxdenom="70000" scalemindenom="1" />
      <rule key="{r2}" label="220 kV station, footprint (OpenStreetMap)" symbol="2" filter="(&quot;max_kv&quot; &lt; 300) AND &quot;source&quot; &lt;&gt; 'box'" scalemaxdenom="70000" scalemindenom="1" />
      <rule key="{r3}" label="220 kV station, box (no footprint)" symbol="3" filter="(&quot;max_kv&quot; &lt; 300) AND &quot;source&quot; = 'box'" scalemaxdenom="70000" scalemindenom="1" />
    </rules>
    <symbols>
      <symbol type="fill" name="0" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="color" type="QString" value="215,25,28,38" />
            <Option name="style" type="QString" value="solid" />
            <Option name="outline_color" type="QString" value="215,25,28,255" />
            <Option name="outline_style" type="QString" value="solid" />
            <Option name="outline_width" type="QString" value="0.5" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="round" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map" />
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="fill" name="1" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="color" type="QString" value="215,25,28,38" />
            <Option name="style" type="QString" value="solid" />
            <Option name="outline_color" type="QString" value="215,25,28,255" />
            <Option name="outline_style" type="QString" value="dash" />
            <Option name="outline_width" type="QString" value="0.5" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="round" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map" />
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="fill" name="2" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="color" type="QString" value="26,150,65,38" />
            <Option name="style" type="QString" value="solid" />
            <Option name="outline_color" type="QString" value="26,150,65,255" />
            <Option name="outline_style" type="QString" value="solid" />
            <Option name="outline_width" type="QString" value="0.5" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="round" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map" />
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
      <symbol type="fill" name="3" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10" is_animated="0">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="color" type="QString" value="26,150,65,38" />
            <Option name="style" type="QString" value="solid" />
            <Option name="outline_color" type="QString" value="26,150,65,255" />
            <Option name="outline_style" type="QString" value="dash" />
            <Option name="outline_width" type="QString" value="0.5" />
            <Option name="outline_width_unit" type="QString" value="MM" />
            <Option name="joinstyle" type="QString" value="round" />
          </Option>
          <data_defined_properties>
            <Option type="Map">
              <Option name="name" type="QString" value="" />
              <Option name="properties" type="Map" />
              <Option name="type" type="QString" value="collection" />
            </Option>
          </data_defined_properties>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <labeling type="rule-based">
    <rules key="{labels-root}">
      <rule key="{label-0}" description="Station name" scalemaxdenom="70000" scalemindenom="1">
        <settings calloutType="simple">
          <text-style fieldName="&quot;name&quot;" isExpression="1" fontFamily="Sans Serif" fontSize="7.0" fontSizeUnit="Point" fontWeight="50" textColor="33,33,33,255" textOpacity="1" fontItalic="0" namedStyle="Regular" multilineHeight="1" allowHtml="0">
            <text-buffer bufferDraw="1" bufferSize="0.8" bufferSizeUnits="MM" bufferColor="255,255,255,230" bufferOpacity="0.9" bufferJoinStyle="128" bufferNoFill="1" />
          </text-style>
          <text-format wrapChar="" autoWrapLength="0" multilineAlign="0" plussign="0" reverseDirectionSymbol="0" />
          <placement placement="8" placementFlags="10" dist="1.2" distUnits="MM" priority="5" repeatDistance="0" overrunDistance="0" offsetType="0" quadOffset="4" centroidInside="1" fitInPolygonOnly="0" />
          <rendering obstacle="1" obstacleFactor="1" obstacleType="1" upsidedownLabels="0" displayAll="0" mergeLines="0" labelPerPart="0" limitNumLabels="0" maxNumLabels="2000" fontLimitPixelSize="0" scaleVisibility="0" drawLabels="1" />
        </settings>
      </rule>
    </rules>
  </labeling>
  <layerGeometryType>2</layerGeometryType>
</qgis>
